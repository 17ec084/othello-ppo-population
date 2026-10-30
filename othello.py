"""Vectorized Othello rules, independent of the learning code.

Coordinates are A1..H1, A2..H8. Black moves first. A pass is compulsory
only when there is no legal move; the game ends when neither side can move.
Bitboard operations adapted from the preceding local neuroevolution experiment.
"""
from __future__ import annotations

import numpy as np

U64 = np.uint64
ALL = U64(0xFFFFFFFFFFFFFFFF)
BITS = np.arange(64, dtype=np.uint64)
DIRECTIONS = ((1, U64(0xFEFEFEFEFEFEFEFE)), (-1, U64(0x7F7F7F7F7F7F7F7F)),
              (8, ALL), (-8, ALL), (9, U64(0xFEFEFEFEFEFEFEFE)),
              (7, U64(0x7F7F7F7F7F7F7F7F)), (-7, U64(0xFEFEFEFEFEFEFEFE)),
              (-9, U64(0x7F7F7F7F7F7F7F7F)))
POWERS = 3 ** np.arange(8, dtype=np.int64)
POSITION = np.array([
    100,-25,10,5,5,10,-25,100, -25,-45,-2,-2,-2,-2,-45,-25,
    10,-2,5,1,1,5,-2,10, 5,-2,1,0,0,1,-2,5,
    5,-2,1,0,0,1,-2,5, 10,-2,5,1,1,5,-2,10,
    -25,-45,-2,-2,-2,-2,-45,-25, 100,-25,10,5,5,10,-25,100,
], dtype=np.float32)


def shift(x, direction):
    n, mask = direction
    return ((x << U64(n)) if n > 0 else (x >> U64(-n))) & mask


def legal_moves(own, opp):
    empty = ~(own | opp)
    result = np.zeros_like(own, dtype=np.uint64)
    for direction in DIRECTIONS:
        ray = shift(own, direction) & opp
        for _ in range(5):
            ray |= shift(ray, direction) & opp
        result |= shift(ray, direction) & empty
    return result


def apply_moves(own, opp, move):
    """Apply legal bitboard moves; callers must validate legality first."""
    flipped = np.zeros_like(own, dtype=np.uint64)
    for direction in DIRECTIONS:
        ray = shift(move, direction) & opp
        captured = ray.copy()
        for _ in range(5):
            ray = shift(ray, direction) & opp
            captured |= ray
        bounded = (shift(captured, direction) & own) != 0
        flipped |= np.where(bounded, captured, U64(0))
    return own | move | flipped, opp & ~flipped


def bits_to_mask(x):
    return ((x[:, None] >> BITS) & U64(1)).astype(bool)


def row_tokens(own, opp):
    """8 base-3 row tokens: empty=0, self=1, opponent=2."""
    cells = bits_to_mask(own).astype(np.int64) + 2 * bits_to_mask(opp)
    return (cells.reshape(-1, 8, 8) * POWERS).sum(-1)


def initial_boards(n):
    return (np.full(n, U64((1 << 28) | (1 << 35))),
            np.full(n, U64((1 << 27) | (1 << 36))))


def stone_counts(boards):
    return bits_to_mask(boards).sum(-1)


def terminal_rewards(black, white, black_agents, agent, mode='win'):
    """Terminal reward from the named agent's perspective."""
    margin = stone_counts(black) - stone_counts(white)
    own_margin = np.where(black_agents == agent, margin, -margin)
    if mode == 'margin':
        return own_margin.astype(np.float32)
    if mode == 'win':
        return (own_margin > 0).astype(np.float32)
    raise ValueError(f'Unknown reward mode: {mode}')
