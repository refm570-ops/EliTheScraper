"""ApprovalGate — sends a proposal card with Confirm/Reject buttons and
executes only on the owner's tap.

The existing AlertDispatcher is send-only (bot.py never reads updates), so this
adds the inbound path: a python-telegram-bot Application that polls for
callback_query updates. It shares the alert bot token; since the dispatcher
never calls getUpdates, there is no polling conflict.

Pending proposals are held in memory with a TTL (the decision's ttl_seconds).
On restart pending approvals are dropped — acceptable, since they expire in
minutes anyway and a dropped proposal simply is not executed (safe default).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import structlog

from trading.models import TradeProposal

log = structlog.get_logger()

ExecuteCallback = Callable[[TradeProposal], Awaitable[None]]


@dataclass
class _Pending:
    proposal: TradeProposal
    message_id: int | None = None


class ApprovalGate:
    def __init__(self, bot_token: str, owner_chat_id: str | int) -> None:
        self._token = bot_token
        try:
            self._owner_chat_id = int(owner_chat_id)
        except (TypeError, ValueError):
            self._owner_chat_id = owner_chat_id  # channel username etc.
        self._pending: dict[str, _Pending] = {}
        self._execute_cb: ExecuteCallback | None = None
        self._app = None

    def set_execute_callback(self, cb: ExecuteCallback) -> None:
        self._execute_cb = cb

    async def start(self) -> None:
        from telegram.ext import Application, CallbackQueryHandler

        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(CallbackQueryHandler(self._on_callback))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        log.info("approval.started")

    async def stop(self) -> None:
        if self._app is None:
            return
        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception:  # noqa: BLE001
            log.warning("approval.stop_error", exc_info=True)
        log.info("approval.stopped")

    async def request(self, proposal: TradeProposal) -> None:
        """Send a proposal card and register it as pending."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        self._prune_expired()
        text = self._format_card(proposal)
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Confirm buy", callback_data=f"tc:{proposal.id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"tr:{proposal.id}"),
        ]])
        msg = await self._app.bot.send_message(
            chat_id=self._owner_chat_id, text=text, parse_mode="HTML",
            reply_markup=keyboard, disable_web_page_preview=True,
        )
        self._pending[proposal.id] = _Pending(proposal=proposal, message_id=msg.message_id)
        log.info("approval.requested", id=proposal.id, ticker=proposal.opportunity.ticker,
                 size_sol=proposal.approved_size_sol)

    async def _on_callback(self, update, context) -> None:  # noqa: ANN001
        query = update.callback_query
        if query is None:
            return
        await query.answer()

        # Only the owner may approve.
        chat = query.message.chat if query.message else None
        if chat is not None and self._owner_chat_id not in (chat.id, getattr(chat, "username", None)):
            log.warning("approval.unauthorized", chat_id=getattr(chat, "id", None))
            return

        data = query.data or ""
        action, _, pid = data.partition(":")
        pending = self._pending.pop(pid, None)

        if pending is None:
            await query.edit_message_text("⚠️ Proposal no longer available (expired or already handled).")
            return
        if pending.proposal.is_expired():
            await query.edit_message_text("⏱️ Proposal expired before confirmation — not executed.")
            return

        if action == "tr":
            await query.edit_message_text(f"❌ Rejected {pending.proposal.opportunity.ticker} — not executed.")
            log.info("approval.rejected", id=pid)
            return

        if action == "tc":
            await query.edit_message_text(
                f"✅ Confirmed {pending.proposal.opportunity.ticker} "
                f"({pending.proposal.approved_size_sol:.3f} SOL) — executing…"
            )
            log.info("approval.confirmed", id=pid)
            if self._execute_cb is not None:
                try:
                    await self._execute_cb(pending.proposal)
                except Exception:  # noqa: BLE001
                    log.error("approval.execute_error", id=pid, exc_info=True)

    def _prune_expired(self) -> None:
        now = time.time()
        expired = [pid for pid, p in self._pending.items() if p.proposal.is_expired(now)]
        for pid in expired:
            self._pending.pop(pid, None)

    @staticmethod
    def _format_card(p: TradeProposal) -> str:
        opp = p.opportunity
        d = p.decision
        soft = p.safety.soft_failures
        soft_line = ""
        if soft:
            soft_line = "\n⚠️ <b>Warnings:</b> " + ", ".join(c.name for c in soft)
        meta = opp.metadata
        return (
            f"🟢 <b>BUY PROPOSAL — {opp.ticker}</b>\n"
            f"<code>{opp.address or 'n/a'}</code>\n\n"
            f"<b>Size:</b> {p.approved_size_sol:.3f} SOL  "
            f"(<b>{d.conviction.value}</b> conviction)\n"
            f"<b>Max slippage:</b> {d.max_slippage_bps/100:.1f}%\n"
            f"<b>Venue:</b> {opp.venue.value} · <b>Source:</b> {opp.source}\n\n"
            f"<b>Liquidity:</b> ${meta.get('liquidity_usd') or 0:,.0f}  "
            f"<b>MCap:</b> ${meta.get('market_cap') or 0:,.0f}\n"
            f"<b>Safety:</b> ✅ passed hard gate{soft_line}\n\n"
            f"<b>Reasoning:</b> {d.reasoning}\n\n"
            f"<i>Expires in {d.ttl_seconds}s</i>"
        )
