import sys, math, pytest
sys.path.insert(0,"/home/fanruochen/CL/RLinf")
from rlinf.workers.actor.fsdp_actor_worker import (
    parse_slot_alt_anneal, slot_alt_schedule_for_step, slot_alt_a_fraction,
    slot_alt_phase,
)

R1 = [[0,"BBBBBBBA"],[9,"BBBBBBBBBBBBBBBA"],[12,"B"]]

def test_none_is_off():
    assert parse_slot_alt_anneal(None) is None
    assert slot_alt_schedule_for_step(None,"BBA",7) == "BBA"

def test_sorted_and_upper():
    st = parse_slot_alt_anneal([[9,"bbba"],[0,"bba"]])
    assert st == [(0,"BBA"),(9,"BBBA")]

def test_stage_selection():
    st = parse_slot_alt_anneal(R1)
    got = [slot_alt_schedule_for_step(st,"BBA",s) for s in range(15)]
    assert got[:9]  == ["BBBBBBBA"]*9
    assert got[9:12]== ["BBBBBBBBBBBBBBBA"]*3
    assert got[12:] == ["B"]*3

def test_unsorted_input_still_selects_correctly():
    st = parse_slot_alt_anneal([[12,"B"],[0,"BBBBBBBA"],[9,"BBBBBBBBBBBBBBBA"]])
    assert slot_alt_schedule_for_step(st,"BBA",11) == "BBBBBBBBBBBBBBBA"
    assert slot_alt_schedule_for_step(st,"BBA",12) == "B"

def test_step_beyond_last_stage_stays_on_last():
    st = parse_slot_alt_anneal(R1)
    assert slot_alt_schedule_for_step(st,"BBA",999) == "B"

def test_unknown_step_falls_back_to_default_not_stage0():
    st = parse_slot_alt_anneal(R1)
    assert slot_alt_schedule_for_step(st,"BBA",None) == "BBA"
    assert slot_alt_schedule_for_step(st,"BBA","?") == "BBA"

def test_null_stage_is_the_joint_ablation():
    st = parse_slot_alt_anneal([[0,"BBA"],[5,None]])
    assert slot_alt_schedule_for_step(st,"BBA",6) is None

def test_a_fraction():
    assert slot_alt_a_fraction("BBA") == pytest.approx(1/3)
    assert slot_alt_a_fraction("BBBBBBBA") == pytest.approx(1/8)
    assert slot_alt_a_fraction("BBBBBBBBBBBBBBBA") == pytest.approx(1/16)
    assert slot_alt_a_fraction("B") == 0.0
    assert math.isnan(slot_alt_a_fraction(None))

@pytest.mark.parametrize("bad", [
    [], "BBA", [[0]], [[0,"BBA","x"]], [[-1,"BBA"]], [[0,"BBA"],[0,"B"]],
    [[1,"BBA"]], [[0,"BXA"]], [[0,""]], [["a","BBA"]],
])
def test_rejects(bad):
    with pytest.raises(ValueError):
        parse_slot_alt_anneal(bad)

def test_pure_B_never_yields_A_and_last_update_still_B():
    pos=0; phases=[]
    for i in range(48):
        ph,pos = slot_alt_phase("B",pos,is_last_update=(i==47)); phases.append(ph)
    assert set(phases)=={"B"} and phases[-1]=="B"

def test_one_in_eight_gives_the_expected_A_count_over_a_step():
    pos=0; a=0
    for i in range(48):
        ph,pos = slot_alt_phase("BBBBBBBA",pos,is_last_update=(i==47))
        a += (ph=="A")
    assert a == 5   # 47 cycled updates -> floor(47/8)=5 A's; the 48th is forced to B
