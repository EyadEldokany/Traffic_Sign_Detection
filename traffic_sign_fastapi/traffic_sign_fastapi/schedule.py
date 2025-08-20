"""
University Timetabling Solver (multi-branch, A/B/C/D groups, VCR vs F2F, labs, capacities)
------------------------------------------------------------------------------------------

This is a production-ready *prototype* using Google OR-Tools (CP-SAT) with a discrete-time model.
It encodes the constraints you provided, including:

- Multi-branch rooms/labs with capacities and types (LAB/ROOM).
- Groups A,B,C,D with inverse F2F/VCR halves of the week.
- Subjects may have 1 or 2 lecture occurrences per week.
- Each lecture occurrence generates *two* cohort lectures: (A,B) and (C,D) that must be on the SAME DAY.
  - Depending on the chosen day, each cohort lecture is either VCR (no room required) or F2F (room required).
- Sections: each subject has exactly **one section per week** per section-group (S1..S6 as per mapping below).
  - Section durations can be 1 or 2 hours.
  - Lab sections must be placed in LAB rooms; other sections can be in ROOM or LAB.
  - Sections must be scheduled **on F2F days** for their group.
- All lectures have duration = 1 hour.
- Resource conflicts prevented: no double-booking rooms/labs, doctors, or same-level students.
- Doctor availability by day (subset of week days they work).
- Capacity constraints: assigned room/lab must meet or exceed the enrolled capacity.

Modeling approach (discrete time):
- Week days D (e.g., Sun..Thu), per-day hourly slots T (e.g., 8..17). Total horizon = D x T.
- For each event e (lecture or section), we create binary variables x[e,d,t,r] for possible placements.
- 2-hour sections are enforced using adjacency: if x[e,d,t,r]==1 then also x[e,d,t+1,r]==1.
- VCR lectures do not require a room variable; we create special x[e,d,t] for VCR-only placements.
- Cohort pair constraint: for each lecture occurrence of a subject, the (AB) and (CD) events must share the same day.

You can replace the sample INPUT block with real data from your DB/API.
Then run `solve()` and inspect the printed schedules per branch, doctor, group, room, and subject.

Requires: pip install ortools
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from ortools.sat.python import cp_model

# =============================
# ---------- INPUT ------------
# =============================

DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu"]  # configurable
START_HOUR = 8
END_HOUR = 18  # exclusive upper bound (i.e., hours: 8..17)
SLOTS_PER_DAY = END_HOUR - START_HOUR

# Face-to-Face patterns (example):
# A,B attend VCR first half (Sun-Tue), F2F second half (Wed-Thu)
# C,D attend F2F first half (Sun-Tue), VCR second half (Wed-Thu)
F2F_DAYS_AB: Set[int] = {3, 4}        # Wed, Thu
F2F_DAYS_CD: Set[int] = {0, 1, 2}     # Sun, Mon, Tue

# Section groups mapping (given):
# For groups (A,C) => S1, S2, S3
# For groups (B,D) => S4, S5, S6
SECTION_GROUPS_AC = ["S1", "S2", "S3"]
SECTION_GROUPS_BD = ["S4", "S5", "S6"]

@dataclass
class Room:
    id: str
    branch: str
    rtype: str  # "LAB" or "ROOM"
    capacity: int

@dataclass
class Doctor:
    id: str
    name: str
    available_days: Set[int]  # set of day indices e.g., {0,1,3}

@dataclass
class Subject:
    id: str
    name: str
    level: str              # e.g., "L1", "L2" to prevent level clashes
    lecture_occurrences: int  # 1 or 2 per week
    lecture_doctor: str
    section_doctor: str
    section_is_lab: bool
    section_duration_hours: int  # 1 or 2
    capacity_ab: int            # enrolled count for A+B lecture
    capacity_cd: int            # enrolled count for C+D lecture
    # Sections capacity per group-bucket (AC vs BD). If you split further per S1..S6, put max per section here.
    capacity_sections_ac: int
    capacity_sections_bd: int

# ------- Sample Data (replace with real DB data) -------
rooms: List[Room] = [
    Room("R1", branch="Main", rtype="ROOM", capacity=120),
    Room("R2", branch="Main", rtype="LAB",  capacity=40),
    Room("R3", branch="Main", rtype="LAB",  capacity=30),
    Room("R4", branch="City", rtype="ROOM", capacity=80),
    Room("R5", branch="City", rtype="LAB",  capacity=35),
]

# Doctors with limited-day availability
doctors: Dict[str, Doctor] = {
    "D_ALG": Doctor("D_ALG", "Dr. Algorithms", {0,1,3}),  # Sun, Mon, Wed
    "D_NET": Doctor("D_NET", "Dr. Networks",   {0,2,3,4}),
    "D_DS":  Doctor("D_DS",  "Dr. DataSci",    {1,2,4}),
}

# Subjects: mixture of 1x and 2x lectures/week
subjects: List[Subject] = [
    Subject(
        id="S-ALG", name="Algorithms", level="L1",
        lecture_occurrences=2, lecture_doctor="D_ALG", section_doctor="D_ALG",
        section_is_lab=False, section_duration_hours=1,
        capacity_ab=100, capacity_cd=90,
        capacity_sections_ac=30, capacity_sections_bd=30,
    ),
    Subject(
        id="S-NET", name="Networks", level="L2",
        lecture_occurrences=1, lecture_doctor="D_NET", section_doctor="D_NET",
        section_is_lab=True, section_duration_hours=2,
        capacity_ab=70, capacity_cd=65,
        capacity_sections_ac=25, capacity_sections_bd=25,
    ),
    Subject(
        id="S-DS", name="Data Science", level="L3",
        lecture_occurrences=1, lecture_doctor="D_DS", section_doctor="D_DS",
        section_is_lab=False, section_duration_hours=2,
        capacity_ab=60, capacity_cd=55,
        capacity_sections_ac=30, capacity_sections_bd=30,
    ),
]

# Assign each subject to a branch for LECTURES & SECTIONS (simplify: all in one branch).
# You can split lectures vs sections per-branch if needed.
subject_branch: Dict[str, str] = {
    "S-ALG": "Main",
    "S-NET": "Main",
    "S-DS":  "City",
}

# =============================
# ------- MODEL BUILD ----------
# =============================

def solve():
    model = cp_model.CpModel()

    # Utility maps
    room_by_id = {r.id: r for r in rooms}
    rooms_by_branch: Dict[str, List[str]] = {}
    for r in rooms:
        rooms_by_branch.setdefault(r.branch, []).append(r.id)

    ALL_DAYS = list(range(len(DAYS)))
    ALL_SLOTS = list(range(START_HOUR, END_HOUR))

    # Data structures to hold variables
    class Event:
        def __init__(self, eid: str, etype: str, subject_id: str, level: str, branch: str,
                     doctor_id: str, group_bucket: str, duration: int, requires_lab: Optional[bool],
                     needs_room: str):
            self.eid = eid
            self.etype = etype              # "LECTURE" or "SECTION"
            self.subject_id = subject_id
            self.level = level
            self.branch = branch
            self.doctor_id = doctor_id
            self.group_bucket = group_bucket  # "AB", "CD", or one of S1..S6
            self.duration = duration          # in hours (1 or 2)
            self.requires_lab = requires_lab  # None for lectures (depends on day), True/False for sections
            self.needs_room = needs_room      # "AUTO" (lecture depends on F2F/VCR), or "YES"/"NO" for sections

    events: List[Event] = []

    # Create lecture events (paired AB/CD per occurrence)
    lecture_pairs: List[Tuple[str, str, str]] = []  # (occurrence_id, eid_AB, eid_CD)

    for subj in subjects:
        for occ in range(subj.lecture_occurrences):
            occ_id = f"{subj.id}_L{occ+1}"
            e_ab = Event(
                eid=f"{occ_id}_AB", etype="LECTURE", subject_id=subj.id, level=subj.level,
                branch=subject_branch[subj.id], doctor_id=subj.lecture_doctor,
                group_bucket="AB", duration=1, requires_lab=None, needs_room="AUTO"
            )
            e_cd = Event(
                eid=f"{occ_id}_CD", etype="LECTURE", subject_id=subj.id, level=subj.level,
                branch=subject_branch[subj.id], doctor_id=subj.lecture_doctor,
                group_bucket="CD", duration=1, requires_lab=None, needs_room="AUTO"
            )
            events.extend([e_ab, e_cd])
            lecture_pairs.append((occ_id, e_ab.eid, e_cd.eid))

    # Create section events (exactly one per S1..S6)
    for subj in subjects:
        for sname in SECTION_GROUPS_AC:  # AC bucket
            events.append(Event(
                eid=f"{subj.id}_{sname}", etype="SECTION", subject_id=subj.id, level=subj.level,
                branch=subject_branch[subj.id], doctor_id=subj.section_doctor,
                group_bucket=sname, duration=subj.section_duration_hours,
                requires_lab=subj.section_is_lab, needs_room="YES"
            ))
        for sname in SECTION_GROUPS_BD:  # BD bucket
            events.append(Event(
                eid=f"{subj.id}_{sname}", etype="SECTION", subject_id=subj.id, level=subj.level,
                branch=subject_branch[subj.id], doctor_id=subj.section_doctor,
                group_bucket=sname, duration=subj.section_duration_hours,
                requires_lab=subj.section_is_lab, needs_room="YES"
            ))

    # Quick lookups
    ev_by_id: Dict[str, Event] = {}
    for e in events:
        ev_by_id[e.eid] = e

    # Variables
    x_f2f: Dict[Tuple[str,int,int,str], cp_model.IntVar] = {}    # (eid, d, h, r) -> BoolVar
    x_vcr: Dict[Tuple[str,int,int], cp_model.IntVar] = {}        # (eid, d, h) -> BoolVar

    def f2f_var(eid, d, h, r):
        v = x_f2f.get((eid, d, h, r))
        if v is None:
            x_f2f[(eid, d, h, r)] = model.NewBoolVar(f"x_f2f[{eid},{d},{h},{r}]")
            v = x_f2f[(eid, d, h, r)]
        return v

    def vcr_var(eid, d, h):
        v = x_vcr.get((eid, d, h))
        if v is None:
            x_vcr[(eid, d, h)] = model.NewBoolVar(f"x_vcr[{eid},{d},{h}]")
            v = x_vcr[(eid, d, h)]
        return v

    # Helper: allowed F2F days for a group bucket
    def group_is_f2f_on_day(bucket: str, d: int) -> bool:
        if bucket in ("AB", "S4", "S5", "S6"):  # B,D bucket follows AB F2F days
            return d in F2F_DAYS_AB
        if bucket in ("CD", "S1", "S2", "S3"):  # A,C bucket follows CD F2F days
            return d in F2F_DAYS_CD
        return False

    # Helper: capacity required
    def capacity_needed(ev: Event) -> int:
        subj = next(s for s in subjects if s.id == ev.subject_id)
        if ev.etype == "LECTURE":
            return subj.capacity_ab if ev.group_bucket == "AB" else subj.capacity_cd
        # Sections: take bucket max (you can refine per S1..S6)
        if ev.group_bucket in ("S1", "S2", "S3"):
            return subj.capacity_sections_ac
        else:
            return subj.capacity_sections_bd

    # Build placement constraints per event
    placed_vars: Dict[str, List[cp_model.IntVar]] = {}  # eid -> list of BoolVars counted in exactly-one

    for ev in events:
        placed_vars[ev.eid] = []
        branch_rooms = rooms_by_branch.get(ev.branch, [])
        doc = doctors[ev.doctor_id]

        # Candidate days = doctor availability only
        cand_days = sorted(list(doc.available_days))

        if ev.etype == "LECTURE":
            # For lectures, needs_room depends on day (F2F vs VCR), always duration 1
            for d in cand_days:
                for h in ALL_SLOTS:
                    if group_is_f2f_on_day(ev.group_bucket, d):
                        # F2F lecture requires a room (any type), capacity satisfied
                        for r in branch_rooms:
                            room = room_by_id[r]
                            if room.capacity < capacity_needed(ev):
                                continue
                            v = f2f_var(ev.eid, d, h, r)
                            placed_vars[ev.eid].append(v)
                    else:
                        # VCR lecture needs no room
                        v = vcr_var(ev.eid, d, h)
                        placed_vars[ev.eid].append(v)
            # Exactly one placement for the lecture
            model.Add(sum(placed_vars[ev.eid]) == 1)

        else:  # SECTION
            # Sections must be on F2F days for their bucket and require rooms; duration may be 1 or 2
            for d in cand_days:
                if not group_is_f2f_on_day(ev.group_bucket, d):
                    continue
                for h in ALL_SLOTS:
                    if ev.duration == 2 and h + 1 >= END_HOUR:
                        continue
                    for r in branch_rooms:
                        room = room_by_id[r]
                        # room type
                        if ev.requires_lab and room.rtype != "LAB":
                            continue
                        if not ev.requires_lab and room.rtype not in ("LAB", "ROOM"):
                            continue
                        # capacity
                        if room.capacity < capacity_needed(ev):
                            continue
                        # add variable(s)
                        v_start = f2f_var(ev.eid, d, h, r)
                        if ev.duration == 1:
                            placed_vars[ev.eid].append(v_start)
                        else:
                            # auxiliary var representing a 2-hour block starting at h in room r
                            v_pair = model.NewBoolVar(f"x_pair[{ev.eid},{d},{h},{r}]")
                            v_next = f2f_var(ev.eid, d, h+1, r)
                            # If v_pair == 1 then both hours are occupied
                            model.Add(v_start == 1).OnlyEnforceIf(v_pair)
                            model.Add(v_next == 1).OnlyEnforceIf(v_pair)
                            placed_vars[ev.eid].append(v_pair)
            # Exactly one placement per section
            model.Add(sum(placed_vars[ev.eid]) == 1)

    # ------------- Conflicts -------------
    # No double-booking in the same room at the same time
    for d in range(len(DAYS)):
        for h in ALL_SLOTS:
            for r in room_by_id.keys():
                same_slot_room_vars = []
                for (eid, dd, hh, rr), var in x_f2f.items():
                    if dd == d and hh == h and rr == r:
                        same_slot_room_vars.append(var)
                if same_slot_room_vars:
                    model.Add(sum(same_slot_room_vars) <= 1)

    # No doctor overlaps (F2F and VCR combined)
    for d in range(len(DAYS)):
        for h in ALL_SLOTS:
            for doc_id in doctors.keys():
                overlap_vars = []
                for ev in events:
                    if ev.doctor_id != doc_id:
                        continue
                    # F2F vars at (d,h)
                    for r in rooms_by_branch.get(ev.branch, []):
                        v = x_f2f.get((ev.eid, d, h, r))
                        if v is not None:
                            overlap_vars.append(v)
                    # VCR vars at (d,h)
                    v = x_vcr.get((ev.eid, d, h))
                    if v is not None:
                        overlap_vars.append(v)
                if overlap_vars:
                    model.Add(sum(overlap_vars) <= 1)

    # No level overlaps (students of the same level cannot attend two events at the same time)
    for d in range(len(DAYS)):
        for h in ALL_SLOTS:
            level_vars: Dict[str, List[cp_model.IntVar]] = {}
            for ev in events:
                # F2F
                for r in rooms_by_branch.get(ev.branch, []):
                    v = x_f2f.get((ev.eid, d, h, r))
                    if v is not None:
                        level_vars.setdefault(ev.level, []).append(v)
                # VCR
                v = x_vcr.get((ev.eid, d, h))
                if v is not None:
                    level_vars.setdefault(ev.level, []).append(v)
            for lvl, vars_list in level_vars.items():
                model.Add(sum(vars_list) <= 1)

    # --------- Lecture pair same-day constraint (FIX: store day indicators) ---------
    # Create and store day indicators for each lecture event (AB or CD) per day.
    day_ind: Dict[Tuple[str,int], cp_model.IntVar] = {}

    def link_day_indicator(eid: str, d: int):
        """Create (or return) a BoolVar that is 1 iff event eid is scheduled on day d."""
        key = (eid, d)
        if key in day_ind:
            return day_ind[key]
        ind = model.NewBoolVar(f"day[{eid},{d}]")
        day_ind[key] = ind

        # Collect all placement vars of eid that are on day d (F2F rooms + VCR slots)
        vars_day: List[cp_model.IntVar] = []
        ev = ev_by_id[eid]
        # F2F placements on day d
        for r in rooms_by_branch.get(ev.branch, []):
            for h in ALL_SLOTS:
                v = x_f2f.get((eid, d, h, r))
                if v is not None:
                    vars_day.append(v)
        # VCR placements on day d
        for h in ALL_SLOTS:
            v = x_vcr.get((eid, d, h))
            if v is not None:
                vars_day.append(v)

        if vars_day:
            # ind == 1  => at least one placement on that day
            model.Add(sum(vars_day) >= 1).OnlyEnforceIf(ind)
            # ind == 0  => no placement on that day
            model.Add(sum(vars_day) == 0).OnlyEnforceIf(ind.Not())
        else:
            # If no candidates that day, force indicator to 0
            model.Add(ind == 0)
        return ind

    # Build indicators and coupling constraints for each lecture occurrence (AB with CD)
    for occ_id, eid_ab, eid_cd in lecture_pairs:
        # Exactly one day for each of AB and CD and same-day equality
        inds_ab = [link_day_indicator(eid_ab, d) for d in range(len(DAYS))]
        inds_cd = [link_day_indicator(eid_cd, d) for d in range(len(DAYS))]
        model.Add(sum(inds_ab) == 1)
        model.Add(sum(inds_cd) == 1)
        # Pairwise equality per day
        for d in range(len(DAYS)):
            model.Add(inds_ab[d] == inds_cd[d])

    # Optional: soft objectives (minimize doctor idle time, compactness, etc.). For prototype, we just search for any feasible solution.

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)

    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No feasible timetable found with the current data and constraints.")
        return

    # --------- Extract & pretty print ---------
    def fmt(d, h):
        return f"{DAYS[d]} {h}:00"

    placements = []  # (branch, when, room/VCR, subject, type, group_bucket, doctor, level)

    # F2F
    for (eid, d, h, r), var in x_f2f.items():
        if solver.Value(var) == 1:
            ev = ev_by_id[eid]
            placements.append((
                ev.branch, fmt(d, h), r, ev.subject_id, ev.etype, ev.group_bucket, ev.doctor_id, ev.level
            ))
    # VCR
    for (eid, d, h), var in x_vcr.items():
        if solver.Value(var) == 1:
            ev = ev_by_id[eid]
            placements.append((
                ev.branch, fmt(d, h), "VCR", ev.subject_id, ev.etype, ev.group_bucket, ev.doctor_id, ev.level
            ))

    # Sort by branch, day/hour, room
    def sort_key(row):
        branch, when, room, *_ = row
        day_name, time = when.split()
        d_idx = DAYS.index(day_name)
        hour = int(time.split(":")[0])
        return (branch, d_idx, hour, room)

    placements.sort(key=sort_key)

    # Print schedule grouped by branch
    from collections import defaultdict
    by_branch = defaultdict(list)
    for row in placements:
        by_branch[row[0]].append(row)

    for branch, rows in by_branch.items():
        print("=== Branch:", branch, "===")
        for row in rows:
            branch, when, room, subj, etype, gb, doc, lvl = row
            print(f"{when:<10} | {room:<4} | {subj:<6} | {etype:<7} | {gb:<3} | {doc:<6} | {lvl}")

if __name__ == "__main__":
    solve()
