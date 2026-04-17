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
        """Convert current facelet state to a CubieCube representation.

        TODO: Implement this method.
        """
        raise NotImplementedError("TODO: implement as_cubie_cube")


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



# ---------------------------------------------------------------------------
# Coordinate encoding/decoding
# ---------------------------------------------------------------------------
# TODO: Implement the following coordinate encoding/decoding functions needed
# by the solver. These should include functions to:
#   - Encode/decode corner orientation (8 corners, orientation 0-2, range 0..2186)
#   - Encode/decode edge orientation (12 edges, orientation 0-1, range 0..2047)
#   - Encode/decode slice edge positions (which 4 of 12 positions hold slice edges 8-11, range 0..494)
#   - Encode corner permutation (8!, range 0..40319)
#   - Encode first-8 edge permutation (8!, range 0..40319)
#   - Encode slice edge permutation (4!, range 0..23)
#   - Calculate permutation parity (0=even, 1=odd)


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
        """Build all move tables and pruning tables needed by the two-phase solver.

        TODO: Implement this method. It should populate:
          - Phase 1 move tables: self.mt_co, self.mt_eo, self.mt_sl
          - Phase 1 pruning tables: self.prun_co_sl, self.prun_eo_sl
          - Phase 2 move tables: self.mt_cp, self.mt_ep8, self.mt_esl
          - Phase 2 pruning tables: self.prun_cp_esl, self.prun_ep8_esl
          - self._p2_mi and self._ready
        """
        raise NotImplementedError("TODO: implement precompute")

    def _dec_perm(self, n: int, size: int) -> List[int]:
        """Decode permutation of given size from Lehmer code index.

        TODO: Implement this method.
        """
        raise NotImplementedError("TODO: implement _dec_perm")

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
