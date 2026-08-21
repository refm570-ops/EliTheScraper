"""Wallet — local, non-custodial keypair for signing Solana transactions.

The secret is loaded from env ONLY (TRADER_WALLET_SECRET, base58) and is never
placed in a log field — structlog renders everything passed to it, so only the
PUBLIC key is ever logged. Signing happens in-process; the key never leaves.

Requires the `trade` extra (solders, base58). Only instantiated in live mode.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger()


class Wallet:
    def __init__(self, secret_base58: str) -> None:
        if not secret_base58:
            raise ValueError("TRADER_WALLET_SECRET is empty; cannot run live trading")
        try:
            import base58
            from solders.keypair import Keypair
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Live trading needs the 'trade' extra: pip install '.[trade]'"
            ) from e

        try:
            raw = base58.b58decode(secret_base58)
            self._keypair = Keypair.from_bytes(raw)
        except Exception as e:
            # Never include the secret in the error.
            raise ValueError("could not parse TRADER_WALLET_SECRET as a base58 keypair") from e

        # Log only the public key, never the secret.
        log.info("wallet.loaded", pubkey=self.pubkey)

    @property
    def pubkey(self) -> str:
        return str(self._keypair.pubkey())

    @property
    def keypair(self):
        """The solders Keypair. Callers sign in-process; do not log this."""
        return self._keypair

    def sign_versioned(self, tx_bytes: bytes):
        """Sign a serialized VersionedTransaction returned by PumpPortal/Jupiter."""
        from solders.transaction import VersionedTransaction

        tx = VersionedTransaction.from_bytes(tx_bytes)
        return VersionedTransaction(tx.message, [self._keypair])
