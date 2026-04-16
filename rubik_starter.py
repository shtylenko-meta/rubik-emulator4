#!/usr/bin/env python3
"""
Rubik's Cube Emulator (3x3) with Two-Phase Autosolve (Kociemba-style).

This file contains:
1. A facelet-based RubiksCube emulator for display/moves
2. A coordinate-based Two-Phase solver built from verified move definitions

Corner numbering (Kociemba standard):
  0:URF  1:UFL  2:ULB  3:UBR  4:DFR  5:DLF  6:DBL  7:DRB

Edge numbering (Kociemba standard):
  0:UR  1:UF  2:UL  3:UB  4:DR  5:DF  6:DL  7:DB  8:FR  9:FL  10:BL  11:BR
"""

import numpy as np
import json
import argparse
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Set, Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_COLORS = {'U': 'W', 'D': 'Y', 'F': 'G', 'B': 'B', 'L': 'O', 'R': 'R'}

# ---------------------------------------------------------------------------
# Move definitions for the solver
# ---------------------------------------------------------------------------
# For each base CW move, define the permutation CYCLE and orientation changes.
# Convention: cycle (a b c d) means a->b->c->d->a
#   i.e., what was at position a moves to position b, etc.
#
# We store these as "where does each piece GO" (image convention):
#   perm[source] = destination
# But for efficiency, we convert to "where does each piece COME FROM":
#   new_state[dest] = old_state[source]   i.e.  new[i] = old[perm_inv[i]]
#
# Actually, let's use the standard Kociemba convention directly:
#   The move is defined as a permutation P where new_cp[i] = old_cp[P[i]]
#   and new_co[i] = (old_co[P[i]] + twist[i]) % 3.
#
# Corner cycles for CW moves:
#   U: (URF UBR ULB UFL) = cycle(0 3 2 1)  -> perm: [3, 0, 1, 2, 4, 5, 6, 7]
#      because new[0] = old[3], new[1] = old[0], new[2] = old[1], new[3] = old[2]
#   R: (URF UBR DRB DFR) = no wait, let me be really careful.
#
# Let me define moves by their cycles, then derive the permutation array.

# Move tables derived from the verified facelet emulator.
# Convention: new[i] = old[perm[i]]  (pull convention)
# new_co[i] = (old_co[perm[i]] + orient[i]) % 3
# new_eo[i] = (old_eo[perm[i]] + orient[i]) % 2

_CORNER_PERM = {
    'U': [1, 2, 3, 0, 4, 5, 6, 7],
    'R': [4, 1, 2, 0, 7, 5, 6, 3],
    'F': [1, 5, 2, 3, 0, 4, 6, 7],
    'D': [0, 1, 2, 3, 7, 4, 5, 6],
    'L': [0, 5, 1, 3, 4, 6, 2, 7],
    'B': [0, 1, 3, 7, 4, 5, 2, 6],
}

_CORNER_ORIENT = {
    'U': [0, 0, 0, 0, 0, 0, 0, 0],
    'R': [2, 0, 0, 1, 1, 0, 0, 2],
    'F': [1, 2, 0, 0, 2, 1, 0, 0],
    'D': [0, 0, 0, 0, 0, 0, 0, 0],
    'L': [0, 1, 2, 0, 0, 2, 1, 0],
    'B': [0, 0, 1, 2, 0, 0, 2, 1],
}

_EDGE_PERM = {
    'U': [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11],
    'R': [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
    'F': [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
    'D': [0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11],
    'L': [0, 1, 9, 3, 4, 5, 10, 7, 8, 6, 2, 11],
    'B': [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
}

_EDGE_ORIENT = {
    'U': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'F': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    'D': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
}

MOVE_NAMES = ['U', "U'", 'U2', 'R', "R'", 'R2', 'F', "F'", 'F2',
              'D', "D'", 'D2', 'L', "L'", 'L2', 'B', "B'", 'B2']
P2_MOVE_NAMES = ['U', "U'", 'U2', 'D', "D'", 'D2', 'R2', 'L2', 'F2', 'B2']


# ---------------------------------------------------------------------------
# Core Emulator Logic (facelet based, for display)
# ---------------------------------------------------------------------------
class RubiksCube:
    def __init__(self) -> None:
        self.faces: Dict[str, np.ndarray] = {f: np.full((3, 3), c, dtype='U1') for f, c in FACE_COLORS.items()}

    def _cycle_rows(self, a: str, ra: int, b: str, rb: int, c: str, rc: int, d: str, rd: int) -> None:
        f = self.faces
        tmp = f[d][rd].copy()
        f[d][rd], f[c][rc], f[b][rb], f[a][ra] = f[c][rc], f[b][rb], f[a][ra], tmp

    def apply(self, move: str) -> None:
        m = move.strip(); face = m[0]; suffix = m[1:]
        times = 3 if suffix == "'" else (2 if suffix == "2" else 1)
        for _ in range(times):
            if face == 'U':
                self.faces['U'] = np.rot90(self.faces['U'], k=1)
                self._cycle_rows('L', 0, 'F', 0, 'R', 0, 'B', 0)
            elif face == 'D':
                self.faces['D'] = np.rot90(self.faces['D'], k=1)
                self._cycle_rows('F', 2, 'L', 2, 'B', 2, 'R', 2)
            elif face == 'F':
                self.faces['F'] = np.rot90(self.faces['F'], k=-1)
                u2 = self.faces['U'][2].copy()
                rc0 = self.faces['R'][:, 0].copy()
                d0 = self.faces['D'][0].copy()
                lc2 = self.faces['L'][:, 2].copy()
                self.faces['U'][2] = lc2[::-1]
                self.faces['R'][:, 0] = u2
                self.faces['D'][0] = rc0[::-1]
                self.faces['L'][:, 2] = d0
            elif face == 'B':
                self.faces['B'] = np.rot90(self.faces['B'], k=-1)
                u0 = self.faces['U'][0].copy()
                lc0 = self.faces['L'][:, 0].copy()
                d2 = self.faces['D'][2].copy()
                rc2 = self.faces['R'][:, 2].copy()
                self.faces['L'][:, 0] = u0[::-1]
                self.faces['D'][2] = lc0
                self.faces['R'][:, 2] = d2[::-1]
                self.faces['U'][0] = rc2
            elif face == 'R':
                self.faces['R'] = np.rot90(self.faces['R'], k=-1)
                fc2 = self.faces['F'][:, 2].copy()
                uc2 = self.faces['U'][:, 2].copy()
                bc0 = self.faces['B'][:, 0].copy()
                dc2 = self.faces['D'][:, 2].copy()
                self.faces['U'][:, 2] = fc2
                self.faces['B'][:, 0] = uc2[::-1]
                self.faces['D'][:, 2] = bc0[::-1]
                self.faces['F'][:, 2] = dc2
            elif face == 'L':
                self.faces['L'] = np.rot90(self.faces['L'], k=1)
                fc0 = self.faces['F'][:, 0].copy()
                uc0 = self.faces['U'][:, 0].copy()
                bc2 = self.faces['B'][:, 2].copy()
                dc0 = self.faces['D'][:, 0].copy()
                self.faces['U'][:, 0] = fc0
                self.faces['B'][:, 2] = uc0[::-1]
                self.faces['D'][:, 0] = bc2[::-1]
                self.faces['F'][:, 0] = dc0

    def is_solved(self) -> bool:
        return all(np.all(f == f[1, 1]) for f in self.faces.values())

    def load_from_dict(self, state: Dict[str, Any]) -> None:
        for fn in self.faces:
            if fn in state:
                self.faces[fn] = np.array(state[fn], dtype='U1')

    def to_dict(self) -> Dict[str, List[List[str]]]:
        return {fn: self.faces[fn].tolist() for fn in self.faces}

    def as_cubie_cube(self) -> 'CubieCube':
        """Convert current facelet state to a CubieCube representation."""
        cc = CubieCube()
        u_col = self.faces['U'][1, 1]
        d_col = self.faces['D'][1, 1]

        # Corners
        for pos in range(8):
            facets = _CORNER_FACETS[pos]
            actual_colors = [self.faces[f][r][c] for f, r, c in facets]
            actual_set = set(actual_colors)
            found = False
            for piece in range(8):
                piece_set = set(_CORNER_COLORS[piece])
                if actual_set == piece_set:
                    cc.cp[pos] = piece
                    for k in range(3):
                        if actual_colors[k] in (u_col, d_col):
                            cc.co[pos] = k
                            break
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not identify corner at position {pos}: {actual_colors}")

        # Edges
        for pos in range(12):
            facets = _EDGE_FACETS[pos]
            actual_colors = [self.faces[f][r][c] for f, r, c in facets]
            actual_set = set(actual_colors)
            found = False
            for piece in range(12):
                piece_set = set(_EDGE_COLORS[piece])
                if actual_set == piece_set:
                    cc.ep[pos] = piece
                    ref_color = _EDGE_COLORS[piece][0]
                    cc.eo[pos] = 0 if actual_colors[0] == ref_color else 1
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not identify edge at position {pos}: {actual_colors}")
        return cc


# ---------------------------------------------------------------------------
# Piece-based cube state (for solver)
# ---------------------------------------------------------------------------
class CubieCube:
    """Stores the cube as corner/edge permutation + orientation arrays."""

    def __init__(self, cp: Optional[List[int]] = None, co: Optional[List[int]] = None,
                 ep: Optional[List[int]] = None, eo: Optional[List[int]] = None) -> None:
        self.cp = cp if cp is not None else list(range(8))
        self.co = co if co is not None else [0] * 8
        self.ep = ep if ep is not None else list(range(12))
        self.eo = eo if eo is not None else [0] * 12

    def copy(self) -> 'CubieCube':
        return CubieCube(self.cp[:], self.co[:], self.ep[:], self.eo[:])

    def apply(self, move_name: str) -> None:
        """Apply a named move (e.g. 'R', "R'", 'R2')."""
        face = move_name[0]
        suffix = move_name[1:] if len(move_name) > 1 else ''
        times = 3 if suffix == "'" else (2 if suffix == '2' else 1)
        cp_f = _CORNER_PERM[face]
        co_f = _CORNER_ORIENT[face]
        ep_f = _EDGE_PERM[face]
        eo_f = _EDGE_ORIENT[face]
        for _ in range(times):
            new_cp = [self.cp[cp_f[i]] for i in range(8)]
            new_co = [(self.co[cp_f[i]] + co_f[i]) % 3 for i in range(8)]
            new_ep = [self.ep[ep_f[i]] for i in range(12)]
            new_eo = [(self.eo[ep_f[i]] + eo_f[i]) % 2 for i in range(12)]
            self.cp, self.co, self.ep, self.eo = new_cp, new_co, new_ep, new_eo

    def is_solved(self) -> bool:
        return (self.cp == list(range(8)) and self.co == [0]*8 and
                self.ep == list(range(12)) and self.eo == [0]*12)


# ---------------------------------------------------------------------------
# Facelet -> CubieCube conversion
# ---------------------------------------------------------------------------
# Corner facelets: for each corner position, the three facelets that belong
# to it. The FIRST facelet is the one on the U or D face (defines orientation 0).
_CORNER_FACETS = [
    [('U',2,2), ('R',0,0), ('F',0,2)],  # 0: URF
    [('U',2,0), ('F',0,0), ('L',0,2)],  # 1: UFL
    [('U',0,0), ('L',0,0), ('B',0,2)],  # 2: ULB
    [('U',0,2), ('B',0,0), ('R',0,2)],  # 3: UBR
    [('D',0,2), ('F',2,2), ('R',2,0)],  # 4: DFR
    [('D',0,0), ('L',2,2), ('F',2,0)],  # 5: DLF
    [('D',2,0), ('B',2,2), ('L',2,0)],  # 6: DBL
    [('D',2,2), ('R',2,2), ('B',2,0)],  # 7: DRB
]

# Edge facelets: for each edge position, the two facelets.
# The FIRST facelet defines orientation 0.
_EDGE_FACETS = [
    [('U',1,2), ('R',0,1)],   # 0: UR
    [('U',2,1), ('F',0,1)],   # 1: UF
    [('U',1,0), ('L',0,1)],   # 2: UL
    [('U',0,1), ('B',0,1)],   # 3: UB
    [('D',1,2), ('R',2,1)],   # 4: DR
    [('D',0,1), ('F',2,1)],   # 5: DF
    [('D',1,0), ('L',2,1)],   # 6: DL
    [('D',2,1), ('B',2,1)],   # 7: DB
    [('F',1,2), ('R',1,0)],   # 8: FR
    [('F',1,0), ('L',1,2)],   # 9: FL
    [('B',1,2), ('L',1,0)],   # 10: BL
    [('B',1,0), ('R',1,2)],   # 11: BR
]

# For each corner, the set of face-colors it contains in the solved state
# Corner i has the colors of the faces its facelets belong to.
# e.g., corner 0 (URF) has colors of U, R, F centers.
_CORNER_COLORS = []  # will be filled by _init_tables
_EDGE_COLORS = []

def _init_color_tables():
    """Precompute which colors belong to each corner/edge in a solved cube."""
    global _CORNER_COLORS, _EDGE_COLORS
    _CORNER_COLORS = []
    for i, facets in enumerate(_CORNER_FACETS):
        colors = tuple(FACE_COLORS[f] for f, r, c in facets)
        _CORNER_COLORS.append(colors)
    _EDGE_COLORS = []
    for i, facets in enumerate(_EDGE_FACETS):
        colors = tuple(FACE_COLORS[f] for f, r, c in facets)
        _EDGE_COLORS.append(colors)

_init_color_tables()


def facelets_to_cubie(faces: Dict[str, np.ndarray]) -> CubieCube:
    """DEPRECATED: Use RubiksCube.as_cubie_cube() instead.
    Convert a facelet dict to a CubieCube.
    """
    temp_cube = RubiksCube()
    temp_cube.faces = faces
    return temp_cube.as_cubie_cube()


# ---------------------------------------------------------------------------
# Coordinate encoding/decoding
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Coordinate encoding/decoding
# ---------------------------------------------------------------------------
def encode_co(co: List[int]) -> int:
    """Encode corner orientations (first 7) as base-3 number. Range: 0..2186."""
    n = 0
    for i in range(7):
        n = n * 3 + co[i]
    return n

def decode_co(n: int) -> List[int]:
    """Decode corner orientation index to 8-element list."""
    co = [0] * 8
    s = 0
    for i in range(6, -1, -1):
        co[i] = n % 3
        n //= 3
        s += co[i]
    co[7] = (3 - s % 3) % 3
    return co

def encode_eo(eo: List[int]) -> int:
    """Encode edge orientations (first 11) as base-2 number. Range: 0..2047."""
    n = 0
    for i in range(11):
        n = n * 2 + eo[i]
    return n

def decode_eo(n: int) -> List[int]:
    """Decode edge orientation index to 12-element list."""
    eo = [0] * 12
    s = 0
    for i in range(10, -1, -1):
        eo[i] = n % 2
        n //= 2
        s += eo[i]
    eo[11] = s % 2  # parity constraint
    return eo

# Slice coordinate: which 4 of the 12 edge positions contain slice edges (8-11)
# We use combinatorial number system to map C(12,4) = 495 combinations.

# Precompute all C(12,4) combinations sorted lexicographically
_SLICE_COMBOS = list(combinations(range(12), 4))
_SLICE_TO_IDX = {c: i for i, c in enumerate(_SLICE_COMBOS)}

def encode_slice(ep: List[int]) -> int:
    """Given edge permutation, find which positions contain slice edges (pieces 8-11).
    Returns index 0..494."""
    positions = tuple(sorted(i for i in range(12) if ep[i] >= 8))
    return _SLICE_TO_IDX[positions]

def decode_slice(idx: int) -> Tuple[int, ...]:
    """Return the 4 positions that contain slice edges for this index."""
    return _SLICE_COMBOS[idx]

# The SOLVED slice state is when slice edges 8,9,10,11 are at positions 8,9,10,11
_SOLVED_SLICE_IDX = _SLICE_TO_IDX[(8, 9, 10, 11)]

# Corner permutation encoding using Lehmer code
def encode_cp(cp: List[int]) -> int:
    """Encode corner permutation using Lehmer code. Range: 0..40319."""
    n = 0
    for i in range(8):
        count = 0
        for j in range(i + 1, 8):
            if cp[j] < cp[i]:
                count += 1
        n = n * (8 - i) + count
    return n

def encode_ep8(ep: List[int]) -> int:
    """Encode the permutation of the first 8 edges using Lehmer code. Range: 0..40319."""
    # Phase 2 only: the first 8 edges stay in first 8 positions
    sub = ep[:8]
    n = 0
    for i in range(8):
        count = 0
        for j in range(i + 1, 8):
            if sub[j] < sub[i]:
                count += 1
        n = n * (8 - i) + count
    return n

def encode_eslice(ep: List[int]) -> int:
    """Encode the permutation of slice edges (positions 8-11). Range: 0..23."""
    sub = [ep[i] - 8 for i in range(8, 12)]  # normalize to 0-3
    n = 0
    for i in range(4):
        count = 0
        for j in range(i + 1, 4):
            if sub[j] < sub[i]:
                count += 1
        n = n * (4 - i) + count
    return n

def get_permutation_parity(p: List[int]) -> int:
    """Calculate the parity of a permutation (0 for even, 1 for odd)."""
    p = list(p)
    parity = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                parity += 1
    return parity % 2


# ---------------------------------------------------------------------------
# Move table precomputation
# ---------------------------------------------------------------------------
class Solver:
    def __init__(self) -> None:
        self._ready: bool = False
        self._solution: List[str] = []
        # Move tables
        self.mt_co: np.ndarray = np.array([])
        self.mt_eo: np.ndarray = np.array([])
        self.mt_sl: np.ndarray = np.array([])
        self.mt_cp: np.ndarray = np.array([])
        self.mt_ep8: np.ndarray = np.array([])
        self.mt_esl: np.ndarray = np.array([])
        # Pruning tables
        self.prun_co_sl: np.ndarray = np.array([])
        self.prun_eo_sl: np.ndarray = np.array([])
        self.prun_cp_esl: np.ndarray = np.array([])
        self.prun_ep8_esl: np.ndarray = np.array([])
        # Phase 2 move indices
        self._p2_mi: List[int] = []

    def precompute(self) -> None:
        if self._ready:
            return

        print("  [autosolve] Precomputing move tables... (this may take a moment)")

        N_CO = 2187
        N_EO = 2048
        N_SL = 495
        N_MOVES = 18

        # Phase 1 move tables
        self.mt_co = np.zeros((N_CO, N_MOVES), dtype=np.int16)
        self.mt_eo = np.zeros((N_EO, N_MOVES), dtype=np.int16)
        self.mt_sl = np.zeros((N_SL, N_MOVES), dtype=np.int16)

        for i in range(N_CO):
            co = decode_co(i)
            for mi, m in enumerate(MOVE_NAMES):
                cc = CubieCube(co=co)
                cc.apply(m)
                self.mt_co[i, mi] = encode_co(cc.co)

        for i in range(N_EO):
            eo = decode_eo(i)
            for mi, m in enumerate(MOVE_NAMES):
                cc = CubieCube(eo=eo)
                cc.apply(m)
                self.mt_eo[i, mi] = encode_eo(cc.eo)

        for i in range(N_SL):
            positions = list(decode_slice(i))
            # Create a dummy EP where slice edges are at these positions
            ep = list(range(12))
            # We need positions[k] to hold a slice edge (>=8)
            # and other positions to hold non-slice edges (<8)
            non_slice_positions = [p for p in range(12) if p not in positions]
            ep_test = [0] * 12
            for k, p in enumerate(positions):
                ep_test[p] = 8 + k  # place slice edges
            ns_val = 0
            for p in range(12):
                if p not in positions:
                    ep_test[p] = ns_val
                    ns_val += 1
            for mi, m in enumerate(MOVE_NAMES):
                cc = CubieCube(ep=ep_test[:])
                cc.apply(m)
                self.mt_sl[i, mi] = encode_slice(cc.ep)

        print("  [autosolve] Building Phase 1 pruning table...")
        # Pruning table: CO x Slice
        self.prun_co_sl = np.full((N_CO, N_SL), 255, dtype=np.uint8)
        self.prun_co_sl[0, _SOLVED_SLICE_IDX] = 0
        q = [(0, _SOLVED_SLICE_IDX)]
        depth = 0
        total = 1
        while q:
            depth += 1
            nq = []
            for co, sl in q:
                for mi in range(N_MOVES):
                    nco = self.mt_co[co, mi]
                    nsl = self.mt_sl[sl, mi]
                    if self.prun_co_sl[nco, nsl] == 255:
                        self.prun_co_sl[nco, nsl] = depth
                        nq.append((nco, nsl))
                        total += 1
            q = nq
            print(f"    CO×SL depth {depth}: {total} states filled")
            if depth >= 12:
                break

        # Pruning table: EO x Slice
        self.prun_eo_sl = np.full((N_EO, N_SL), 255, dtype=np.uint8)
        self.prun_eo_sl[0, _SOLVED_SLICE_IDX] = 0
        q = [(0, _SOLVED_SLICE_IDX)]
        depth = 0
        total = 1
        while q:
            depth += 1
            nq = []
            for eo, sl in q:
                for mi in range(N_MOVES):
                    neo = self.mt_eo[eo, mi]
                    nsl = self.mt_sl[sl, mi]
                    if self.prun_eo_sl[neo, nsl] == 255:
                        self.prun_eo_sl[neo, nsl] = depth
                        nq.append((neo, nsl))
                        total += 1
            q = nq
            print(f"    EO×SL depth {depth}: {total} states filled")
            if depth >= 12:
                break

        # Phase 2 move tables: corner permutation under P2 moves only
        print("  [autosolve] Building Phase 2 tables...")
        N_CP = 40320  # 8!
        N_EP8 = 40320 # 8!
        N_ESL = 24    # 4!
        N_P2 = len(P2_MOVE_NAMES)  # 10
        self._p2_mi = [MOVE_NAMES.index(m) for m in P2_MOVE_NAMES]

        self.mt_cp = np.zeros((N_CP, N_P2), dtype=np.int32)
        self.mt_ep8 = np.zeros((N_EP8, N_P2), dtype=np.int32)
        self.mt_esl = np.zeros((N_ESL, N_P2), dtype=np.int8)

        for i in range(N_CP):
            cp = self._dec_perm(i, 8)
            for pi, m in enumerate(P2_MOVE_NAMES):
                cc = CubieCube(cp=cp[:])
                cc.apply(m)
                self.mt_cp[i, pi] = encode_cp(cc.cp)

        for i in range(N_EP8):
            ep = self._dec_perm(i, 8) + [8, 9, 10, 11]
            for pi, m in enumerate(P2_MOVE_NAMES):
                cc = CubieCube(ep=ep[:])
                cc.apply(m)
                self.mt_ep8[i, pi] = encode_ep8(cc.ep)

        for i in range(N_ESL):
            ep = [0, 1, 2, 3, 4, 5, 6, 7] + [x + 8 for x in self._dec_perm(i, 4)]
            for pi, m in enumerate(P2_MOVE_NAMES):
                cc = CubieCube(ep=ep[:])
                cc.apply(m)
                self.mt_esl[i, pi] = encode_eslice(cc.ep)

        # Phase 2 pruning: CP x SlicePerm and EP8 x SlicePerm
        print("  [autosolve] Building Phase 2 pruning tables...")
        self.prun_cp_esl = np.full((N_CP, N_ESL), 255, dtype=np.uint8)
        self.prun_cp_esl[0, 0] = 0
        q = [(0, 0)]
        depth = 0
        total = 1
        while q:
            depth += 1
            nq = []
            for cp, esl in q:
                for pi in range(N_P2):
                    ncp = self.mt_cp[cp, pi]
                    nesl = self.mt_esl[esl, pi]
                    if self.prun_cp_esl[ncp, nesl] == 255:
                        self.prun_cp_esl[ncp, nesl] = depth
                        nq.append((ncp, nesl))
                        total += 1
            q = nq
            if depth >= 14: break


        self.prun_ep8_esl = np.full((N_EP8, N_ESL), 255, dtype=np.uint8)
        self.prun_ep8_esl[0, 0] = 0
        q = [(0, 0)]
        depth = 0
        total = 1
        while q:
            depth += 1
            nq = []
            for ep, esl in q:
                for pi in range(N_P2):
                    nep = self.mt_ep8[ep, pi]
                    nesl = self.mt_esl[esl, pi]
                    if self.prun_ep8_esl[nep, nesl] == 255:
                        self.prun_ep8_esl[nep, nesl] = depth
                        nq.append((nep, nesl))
                        total += 1
            q = nq
            if depth >= 14: break

        self._ready = True
        print("  [autosolve] Tables ready.")

    def _dec_perm(self, n: int, size: int) -> List[int]:
        """Decode permutation of given size from Lehmer code index."""
        p = [0] * size
        digits = []
        for i in range(size):
            digits.append(n % (i + 1))
            n //= (i + 1)
        digits.reverse()
        available = list(range(size))
        for i in range(size):
            p[i] = available.pop(digits[i])
        return p

    def solve(self, cube: RubiksCube) -> List[str]:
        """Solve using Two-Phase IDA*. Returns list of move strings.

        TODO: Implement this method.
        """
        raise NotImplementedError("TODO: implement the solve method")

    def _phase1_search(self, cc: CubieCube, depth: int, last_move: int) -> bool:
        """Phase 1 IDA* search. Returns True if solution found (stored in self._solution).

        TODO: Implement this method.
        """
        raise NotImplementedError("TODO: implement Phase 1 search")

    def _phase2(self, cc: CubieCube, max_depth: int) -> bool:
        """Phase 2 entry point: solve remaining permutation with restricted moves.

        TODO: Implement this method.
        """
        raise NotImplementedError("TODO: implement Phase 2 entry")

    def _phase2_search(self, cc_unused: Optional[CubieCube], depth: int, last_move: int,
                       cp_idx: int, ep8_idx: int, esl_idx: int) -> Optional[List[str]]:
        """Phase 2 IDA* search using coordinate-based move tables.

        TODO: Implement this method.
        """
        raise NotImplementedError("TODO: implement Phase 2 search")


_SOLVER = Solver()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True, help='Path to save the solution JSON')
    args = parser.parse_args()

    with open(args.input, 'r') as fh:
        data = json.load(fh)

    cube = RubiksCube()
    cube.load_from_dict(data)

    starting_pos = cube.to_dict()
    steps = []

    print(f"\nSolving: {args.input}")

    if cube.is_solved():
        print("  Already solved!")
    else:
        solution = _SOLVER.solve(cube)
        if solution:
            print(f"  Solution ({len(solution)} moves): {' '.join(solution)}")
            for mv in solution:
                cube.apply(mv)
                steps.append(mv)
                if cube.is_solved():
                    print("  SOLVED!")
                    break
        else:
            print("  No solution found.")

    print(f"  Final state: {'SOLVED' if cube.is_solved() else 'Not solved'}")

    output_data = {
        "starting_position": starting_pos,
        "steps": steps,
        "final_position": cube.to_dict()
    }
    with open(args.output, 'w') as fh:
        json.dump(output_data, fh, indent=2)
    print(f"  Result saved -> {args.output}")


if __name__ == '__main__':
    main()
