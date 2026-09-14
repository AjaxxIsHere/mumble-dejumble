"""Rambler Stress Test — 200 HAND-WRITTEN records (the final hold-out).

These are authored by hand, never generated from the training templates, and
must NEVER be used for training or tuning (spec §10-11, §23). Ids use a
disjoint `stress_` prefix and tests assert no overlap with any split.

Record content is authored compactly as (input, output, annotations); the
finalizer derives entity metadata from the clean output and fills correction
kind/marker details, so hand-authoring stays about CONTENT.

Levels: 50 easy (simple cleanup), 50 medium (multiple disfluencies),
50 hard (long + multiple corrections), 50 extreme (long + corrections +
formatting + many entities, including combinations NOT in the training matrix).
"""

import hashlib
from pathlib import Path

from data.common import write_jsonl
from data.generators.composer import entities_from_text
from data.generators.entities import entity_spans

LEVELS = ("easy", "medium", "hard", "extreme")
EXPECTED_PER_LEVEL = 50


class StressError(Exception):
    pass


def _c(
    in_, out, cat, tr=(), corr=(), reform=(), fmt=None, pres=False, preserved=(),
    domain="everyday", notes=None,
):
    """Compact record constructor: (input, output, category, ...)."""
    return {
        "in": in_, "out": out, "cat": cat, "tr": tuple(tr),
        "corr": tuple(corr), "reform": tuple(reform), "fmt": fmt,
        "pres": pres, "preserved": tuple(preserved), "domain": domain, "notes": notes,
    }


# --------------------------------------------------------------------------
# EASY — simple cleanup (50)
# --------------------------------------------------------------------------

EASY = [
    # fillers only
    _c("uh I want coffee", "I want coffee.", "basic", ("filler",)),
    _c("um basically I think we should go", "I think we should go.", "filler", ("filler",)),
    _c("so like we need to talk", "We need to talk.", "filler", ("filler",)),
    _c("I'm kind of sort of tired", "I'm tired.", "filler", ("filler",)),
    _c("you know what I mean it's late", "It's late.", "filler", ("filler",)),
    _c("well um okay let's start", "Let's start.", "filler", ("filler",)),
    _c("the meeting is at um three", "The meeting is at three.", "filler", ("filler",)),
    _c("I guess we should leave now", "We should leave now.", "filler", ("filler",)),
    _c("basically er the server is down", "The server is down.", "filler", ("filler",)),
    _c("so yeah we're ready", "We're ready.", "filler", ("filler",)),
    _c("hmm like the food was good", "The food was good.", "filler", ("filler",)),
    _c("ah I think it's Tuesday", "I think it's Tuesday.", "filler", ("filler",)),
    _c("right so the plan is set", "The plan is set.", "filler", ("filler",)),
    _c("okay so um where is the file", "Where is the file?", "filler", ("filler",)),
    # word stutters only
    _c("I I want to go to the the mall", "I want to go to the mall.", "repetition", ("repetition",)),
    _c("we we should start soon", "We should start soon.", "repetition", ("repetition",)),
    _c("send send the file please", "Send the file please.", "repetition", ("repetition",)),
    _c("the the report is ready", "The report is ready.", "repetition", ("repetition",)),
    _c("it's it's not working", "It's not working.", "repetition", ("repetition",)),
    _c("can we we meet now", "Can we meet now?", "repetition", ("repetition",)),
    # single corrections
    _c("Tuesday no Thursday", "Thursday.", "correction", ("self_correction",),
      corr=(("Tuesday", "Thursday"),)),
    _c("send it to John no Sarah", "Send it to Sarah.", "correction", ("self_correction",),
      corr=(("John", "Sarah"),)),
    _c("port 8080 no 3000", "Port 3000.", "correction", ("self_correction",),
      corr=(("8080", "3000"),)),
    _c("3 PM no actually 4:30", "4:30 PM.", "correction", ("self_correction",),
      corr=(("3 PM", "4:30 PM"),)),
    _c("version 2 no version 3", "Version 3.", "correction", ("self_correction",),
      corr=(("version 2", "version 3"),)),
    _c("Dubai no Abu Dhabi", "Abu Dhabi.", "correction", ("self_correction",),
      corr=(("Dubai", "Abu Dhabi"),)),
    _c("call me tomorrow actually tonight", "Call me tonight.", "correction", ("self_correction",),
      corr=(("tomorrow", "tonight"),)),
    _c("I'll take the 5 PM no the 6 PM train", "I'll take the 6 PM train.", "correction",
      ("self_correction",), corr=(("5 PM", "6 PM"),)),
    # punctuation / capitalization only (no disfluencies)
    _c("send the files to sarah on friday", "Send the files to Sarah on Friday.",
      "basic", ()),
    _c("we should meet at 3 pm tomorrow", "We should meet at 3 PM tomorrow.", "basic", ()),
    _c("the meeting is on tuesday in dubai", "The meeting is on Tuesday in Dubai.",
      "basic", ()),
    _c("is the report ready", "Is the report ready?", "basic", ()),
    _c("i need 2 copies", "I need 2 copies.", "basic", ()),
    _c("john called about the contract", "John called about the contract.", "basic", ()),
    _c("dinner at 7", "Dinner at 7.", "basic", ()),
    _c("call the client back", "Call the client back.", "basic", ()),
    _c("the server is running", "The server is running.", "basic", ()),
    _c("pack the charger for the trip", "Pack the charger for the trip.", "basic", ()),
    # already clean — nothing to do
    _c("Please remind Omar about the call.", "Please remind Omar about the call.", "basic", ()),
    _c("The report is due on Monday.", "The report is due on Monday.", "basic", ()),
    _c("We should review the report before Friday.", "We should review the report before Friday.",
      "basic", ()),
    _c("I'll call you tomorrow.", "I'll call you tomorrow.", "basic", ()),
    _c("The meeting is at 3 PM.", "The meeting is at 3 PM.", "basic", ()),
    _c("Please send the invoice to Maria.", "Please send the invoice to Maria.", "basic", ()),
    _c("Let's meet at the cafe.", "Let's meet at the cafe.", "basic", ()),
    _c("The train leaves at 8.", "The train leaves at 8.", "basic", ()),
    _c("I need to buy milk, eggs, and bread.", "I need to buy milk, eggs, and bread.",
      "basic", (), pres=True, preserved=("milk, eggs, and bread",)),
    _c("The meeting is from 3 to 4.", "The meeting is from 3 to 4.", "basic", (),
      pres=True, preserved=("from 3 to 4",)),
    _c("I was going to say Tuesday, but Wednesday works better.",
      "I was going to say Tuesday, but Wednesday works better.", "basic", (),
      pres=True, preserved=("I was going to say Tuesday, but Wednesday works better",)),
    _c("The temperature stays between 20 and 25 degrees.",
      "The temperature stays between 20 and 25 degrees.", "basic", (),
      pres=True, preserved=("between 20 and 25",)),
]

# --------------------------------------------------------------------------
# MEDIUM — multiple disfluencies (50)
# --------------------------------------------------------------------------

MEDIUM = [
    # filler + repetition (8)
    _c("um I I wanted to ask about the the budget", "I wanted to ask about the budget.",
      "mixed", ("filler", "repetition"), domain="work"),
    _c("so uh we we should reschedule the the call", "We should reschedule the call.",
      "mixed", ("filler", "repetition"), domain="scheduling"),
    _c("you know I I can't find the the keys", "I can't find the keys.",
      "mixed", ("filler", "repetition"), domain="everyday"),
    _c("like the the store was was closed", "The store was closed.",
      "mixed", ("filler", "repetition"), domain="shopping"),
    _c("basically um we're we're out of time", "We're out of time.",
      "mixed", ("filler", "repetition"), domain="work"),
    _c("er I I need to to finish this today", "I need to finish this today.",
      "mixed", ("filler", "repetition"), domain="work"),
    _c("hmm the the wifi keeps keeps dropping", "The wifi keeps dropping.",
      "mixed", ("filler", "repetition"), domain="technical"),
    _c("well um he he said it's it's fine", "He said it's fine.",
      "mixed", ("filler", "repetition"), domain="everyday"),
    # filler + correction (8)
    _c("uh can you send it to John actually no Sarah", "Can you send it to Sarah?",
      "mixed", ("filler", "self_correction"), corr=(("John", "Sarah"),), domain="messaging"),
    _c("so like meet me at 3 wait 4:30", "Meet me at 4:30.",
      "mixed", ("filler", "self_correction"), corr=(("3", "4:30"),), domain="scheduling"),
    _c("um the deadline is Friday I mean Monday", "The deadline is Monday.",
      "mixed", ("filler", "self_correction"), corr=(("Friday", "Monday"),), domain="work"),
    _c("basically we're going to Dubai no Abu Dhabi", "We're going to Abu Dhabi.",
      "mixed", ("filler", "self_correction"), corr=(("Dubai", "Abu Dhabi"),), domain="travel"),
    _c("you know the price is 50 no 45 dollars", "The price is 45 dollars.",
      "mixed", ("filler", "self_correction"), corr=(("50", "45"),), domain="shopping"),
    _c("err port 8080 sorry port 9090", "Port 9090.",
      "mixed", ("filler", "self_correction"), corr=(("8080", "9090"),), domain="technical"),
    _c("well I'll come Tuesday actually Wednesday", "I'll come Wednesday.",
      "mixed", ("filler", "self_correction"), corr=(("Tuesday", "Wednesday"),), domain="scheduling"),
    _c("ah give me two no three copies", "Give me three copies.",
      "mixed", ("filler", "self_correction"), corr=(("two", "three"),), domain="work"),
    # repetition + correction (8)
    _c("I I need the the report by Thursday no Friday",
      "I need the report by Friday.", "mixed", ("repetition", "self_correction"),
      corr=(("Thursday", "Friday"),), domain="work"),
    _c("the the meeting is is at 2 PM no wait 3 PM",
      "The meeting is at 3 PM.", "mixed", ("repetition", "self_correction"),
      corr=(("2 PM", "3 PM"),), domain="scheduling"),
    _c("call call Priya no wait Aisha tomorrow",
      "Call Aisha tomorrow.", "mixed", ("repetition", "self_correction"),
      corr=(("Priya", "Aisha"),), domain="messaging"),
    _c("send send the files to to the office no the lab",
      "Send the files to the lab.", "mixed", ("repetition", "self_correction"),
      corr=(("the office", "the lab"),), domain="work"),
    _c("we we fly on on Monday actually Tuesday",
      "We fly on Tuesday.", "mixed", ("repetition", "self_correction"),
      corr=(("Monday", "Tuesday"),), domain="travel"),
    _c("it it costs 20 no 25 dollars",
      "It costs 25 dollars.", "mixed", ("repetition", "self_correction"),
      corr=(("20", "25"),), domain="shopping"),
    _c("the the server runs on on port 8080 wait port 3000",
      "The server runs on port 3000.", "mixed", ("repetition", "self_correction"),
      corr=(("8080", "3000"),), domain="technical"),
    _c("I'm I'm meeting Omar no no Hana at 5",
      "I'm meeting Hana at 5.", "mixed", ("repetition", "self_correction"),
      corr=(("Omar", "Hana"),), domain="scheduling"),
    # reformulation (8)
    _c("I wanted to say that maybe we could go over the numbers—actually let's just meet tomorrow.",
      "Let's just meet tomorrow.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "I wanted to say that maybe we could go over the numbers", "kept": "Let's just meet tomorrow."},)),
    _c("what I meant to say was we should maybe review the slides, anyway, we should ship on Friday.",
      "We should ship on Friday.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "what I meant to say was we should maybe review the slides", "kept": "We should ship on Friday."},)),
    _c("so the thing is I was thinking about the budget, actually, send the invoice today.",
      "Send the invoice today.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "so the thing is I was thinking about the budget", "kept": "Send the invoice today."},)),
    _c("I'll send the, wait, I'll send the file to Sarah.",
      "I'll send the file to Sarah.", "reformulation", ("reformulation",),
      reform=({"mode": "restart", "abandoned": "I'll send the", "kept": "I'll send the file to Sarah."},)),
    _c("we should, I mean, we should book the tickets tonight.",
      "We should book the tickets tonight.", "reformulation", ("reformulation",),
      reform=({"mode": "restart", "abandoned": "we should", "kept": "We should book the tickets tonight."},)),
    _c("hmm let me rephrase that, the meeting is on hold.",
      "The meeting is on hold.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "hmm let me rephrase that", "kept": "The meeting is on hold."},)),
    _c("you know what I wanted to say that maybe we could wait, so actually, call them now.",
      "Call them now.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "you know what I wanted to say that maybe we could wait", "kept": "Call them now."},)),
    _c("can you, what I meant is, can you bring the charger.",
      "Can you bring the charger?", "reformulation", ("reformulation",),
      reform=({"mode": "restart", "abandoned": "can you", "kept": "Can you bring the charger?"},)),
    # preservation with noise (6)
    _c("uh the meeting runs from 3 PM to 4 PM",
      "The meeting runs from 3 PM to 4 PM.", "repetition", ("filler",), pres=True,
      preserved=("from 3 PM to 4 PM",), domain="scheduling"),
    _c("um we will review the report between Monday and Friday",
      "We will review the report between Monday and Friday.", "repetition", ("filler",),
      pres=True, preserved=("between Monday and Friday",), domain="work"),
    _c("uh it was really, really important to fix the database",
      "It was really, really important to fix the database.", "repetition", ("filler",),
      pres=True, preserved=("really, really",), domain="technical"),
    _c("so the gate code is 4829, don't change it",
      "The gate code is 4829, don't change it.", "repetition", ("filler",),
      pres=True, preserved=("The gate code is 4829",), domain="everyday"),
    _c("um call me before 5 PM or after 7 PM",
      "Call me before 5 PM or after 7 PM.", "repetition", ("filler",),
      pres=True, preserved=("before 5 PM or after 7 PM",), domain="messaging"),
    _c("like I'll say it again: the meeting is on Tuesday at 3 PM",
      "I'll say it again: the meeting is on Tuesday at 3 PM.", "repetition", ("filler",),
      pres=True, preserved=("I'll say it again",), domain="scheduling"),
    # simple formatting (12)
    _c("make a list of milk eggs and bread", "- Milk\n- Eggs\n- Bread",
      "formatting", ("formatting",), fmt={"type": "bullet_list", "items": ["milk", "eggs", "bread"]},
      domain="lists"),
    _c("put apples bananas and oranges in a bulleted list",
      "- Apples\n- Bananas\n- Oranges", "formatting", ("formatting",),
      fmt={"type": "bullet_list", "items": ["apples", "bananas", "oranges"]}, domain="lists"),
    _c("make me a list of coffee and tea", "- Coffee\n- Tea", "formatting", ("formatting",),
      fmt={"type": "bullet_list", "items": ["coffee", "tea"]}, domain="lists"),
    _c("add sugar flour and rice to a list", "1. Sugar\n2. Flour\n3. Rice",
      "formatting", ("formatting",),
      fmt={"type": "bullet_list", "items": ["sugar", "flour", "rice"]}, domain="lists"),
    _c("make this a heading: notes from the Tuesday call",
      "# Notes from the Tuesday call", "formatting", ("formatting",),
      fmt={"type": "heading", "items": ["notes from the Tuesday call"]}, domain="notes"),
    _c("use the Kubernetes migration plan as a heading",
      "# The Kubernetes migration plan", "formatting", ("formatting",),
      fmt={"type": "heading", "items": ["the Kubernetes migration plan"]}, domain="technical"),
    _c("turn this into a title: trip to Lisbon",
      "# Trip to Lisbon", "formatting", ("formatting",),
      fmt={"type": "heading", "items": ["trip to Lisbon"]}, domain="travel"),
    _c("make it polite: Send me the report by Friday.",
      "Could you please send me the report by Friday?", "formatting", ("formatting",),
      fmt={"type": "polite_message", "items": ["Send me the report by Friday."]}, domain="email"),
    _c("write a polite version of Call Priya back about the invoice.",
      "I'd appreciate it if you could call Priya back about the invoice.", "formatting",
      ("formatting",), fmt={"type": "polite_message", "items": ["Call Priya back about the invoice."]},
      domain="email"),
    _c("can you make this sound polite: Book the conference room for Monday.",
      "When you get a chance, could you book the conference room for Monday?", "formatting",
      ("formatting",), fmt={"type": "polite_message", "items": ["Book the conference room for Monday."]},
      domain="work"),
    _c("polish this up: Confirm the 4 PM slot.",
      "Please confirm the 4 PM slot when you can.", "formatting", ("formatting",),
      fmt={"type": "polite_message", "items": ["Confirm the 4 PM slot."]}, domain="scheduling"),
    _c("make this a heading", "# Heading", "formatting", ("formatting",),
      fmt={"type": "heading", "items": ["Heading"]}, domain="notes"),
]

# --------------------------------------------------------------------------
# HARD — long speech + multiple corrections (50)
# --------------------------------------------------------------------------

HARD = [
    # long + 2 corrections (12)
    _c("so I was talking to the client earlier and they want the files by Friday no wait Monday and also the price changed from 500 no 450 dollars",
      "I was talking to the client earlier and they want the files by Monday and also the price changed from 450 dollars.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("Friday", "Monday"), ("500", "450")), domain="work"),
    _c("okay so the meeting with Priya is at 2 PM no sorry 3 PM and we should book the conference room no wait the lab",
      "The meeting with Priya is at 3 PM and we should book the lab.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("2 PM", "3 PM"), ("the conference room", "the lab")), domain="scheduling"),
    _c("I need you to deploy the update to staging no wait production and the version should be 2.4 no 2.5",
      "I need you to deploy the update to production and the version should be 2.5.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("staging", "production"), ("2.4", "2.5")), domain="technical"),
    _c("we're flying to Dubai no actually Abu Dhabi and the return date is Sunday no Monday",
      "We're flying to Abu Dhabi and the return date is Monday.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("Dubai", "Abu Dhabi"), ("Sunday", "Monday")), domain="travel"),
    _c("she said the budget meeting is on Tuesday no Wednesday and it starts at 9 no 9:30",
      "She said the budget meeting is on Wednesday and it starts at 9:30.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("Tuesday", "Wednesday"), ("9", "9:30")), domain="work"),
    _c("the server went down because of the database no the cache layer and we need to restart it at midnight no at 1 AM",
      "The server went down because of the cache layer and we need to restart it at 1 AM.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("the database", "the cache layer"), ("midnight", "1 AM")), domain="technical"),
    _c("I'll pick up the kids at 5 no 5:30 and then we're going to the park no the beach",
      "I'll pick up the kids at 5:30 and then we're going to the beach.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("5", "5:30"), ("the park", "the beach")), domain="everyday"),
    _c("the report goes to Ahmed no Omar and it's due on the fifth no the seventh of March",
      "The report goes to Omar and it's due on the seventh of March.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("Ahmed", "Omar"), ("the fifth", "the seventh")), domain="work"),
    _c("we should order three no four of the printers and ship them to Berlin no Munich",
      "We should order four of the printers and ship them to Munich.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("three", "four"), ("Berlin", "Munich")), domain="shopping"),
    _c("the call with the vendor is at 4 PM no 4:30 and we'll use Teams no Zoom",
      "The call with the vendor is at 4:30 and we'll use Zoom.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("4 PM", "4:30"), ("Teams", "Zoom")), domain="work"),
    _c("I booked the hotel for two nights no three and the check-in is at 2 no at 3",
      "I booked the hotel for three nights and the check-in is at 3.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("two", "three"), ("2", "3")), domain="travel"),
    _c("send the survey to the team by Thursday no Friday and remind Tariq no Nadia about the deadline",
      "Send the survey to the team by Friday and remind Nadia about the deadline.",
      "mixed", ("self_correction", "self_correction"),
      corr=(("Thursday", "Friday"), ("Tariq", "Nadia")), domain="work"),
    # long + filler + repetition + correction (12)
    _c("uh basically I was talking to John yesterday and he said that we could probably have the meeting on Tuesday morning but actually I just remembered that he's travelling Tuesday so can you change it to Wednesday afternoon instead",
      "John said we could have the meeting Tuesday morning, but he's travelling Tuesday, so can you change it to Wednesday afternoon instead?",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("Tuesday morning", "Wednesday afternoon"),), domain="scheduling", notes="spec §4 long example"),
    _c("so uh we we need to review the the quarterly numbers before the board meeting and I I think the deadline is Friday no wait Thursday",
      "We need to review the quarterly numbers before the board meeting and I think the deadline is Thursday.",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("Friday", "Thursday"),), domain="work"),
    _c("um you know I I wanted to ask if we can can move the the workshop to the afternoon because the the morning is booked",
      "I wanted to ask if we can move the workshop to the afternoon because the morning is booked.",
      "mixed", ("filler", "repetition"), domain="work"),
    _c("okay so basically um the the client rejected the the first draft and they they want us to redo the the intro by tomorrow no Friday",
      "The client rejected the first draft and they want us to redo the intro by Friday.",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("tomorrow", "Friday"),), domain="work"),
    _c("um we we should probably order more of the the packaging before the the sale starts and I I think we need two hundred no three hundred units",
      "We should probably order more of the packaging before the sale starts and I think we need three hundred units.",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("two hundred", "three hundred"),), domain="shopping"),
    _c("so uh the the database backup is is scheduled for tonight but but the the team wants it on the weekend instead",
      "The database backup is scheduled for tonight, but the team wants it on the weekend instead.",
      "mixed", ("filler", "repetition"), domain="technical"),
    _c("you know um I I was looking at the the schedule and I I realized the the trip overlaps with the the conference so so we should push it to next month",
      "I was looking at the schedule and I realized the trip overlaps with the conference, so we should push it to next month.",
      "mixed", ("filler", "repetition"), domain="travel"),
    _c("basically um we we ran out of of toner for for the printer and we we need it before before the 3 PM meeting no wait the 2 PM meeting",
      "We ran out of toner for the printer and we need it before the 2 PM meeting.",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("3 PM", "2 PM"),), domain="work"),
    _c("ah um the the wifi in in the conference room keeps keeps cutting out and and I I think it's the the router no the modem",
      "The wifi in the conference room keeps cutting out and I think it's the modem.",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("the router", "the modem"),), domain="technical"),
    _c("well um I I finally heard back from from the insurance company and and they they said the the claim is approved",
      "I finally heard back from the insurance company and they said the claim is approved.",
      "mixed", ("filler", "repetition"), domain="everyday"),
    _c("okay so so for the the offsite um we we should pick the the venue by by Wednesday no Thursday and and send out the the invites",
      "For the offsite we should pick the venue by Thursday and send out the invites.",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("Wednesday", "Thursday"),), domain="work"),
    _c("um I I was thinking about the the redesign and and I I actually prefer the the blue version no wait the green version",
      "I was thinking about the redesign and I actually prefer the green version.",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("the blue version", "the green version"),), domain="work"),
    # adversarial preservation — corrections that must NOT be resolved (6)
    _c("I was going to say Tuesday, but Wednesday works better.",
      "I was going to say Tuesday, but Wednesday works better.", "basic", (), pres=True,
      preserved=("I was going to say Tuesday, but Wednesday works better",), notes="adversarial: both dates meaningful"),
    _c("the old price was 50 and the new price is 45",
      "The old price was 50 and the new price is 45.", "basic", (), pres=True,
      preserved=("The old price was 50",)),
    _c("we compared version 2 and version 3",
      "We compared version 2 and version 3.", "basic", (), pres=True,
      preserved=("version 2 and version 3",)),
    _c("the flight departs at 6 AM and arrives at 9 AM",
      "The flight departs at 6 AM and arrives at 9 AM.", "basic", (), pres=True,
      preserved=("departs at 6 AM and arrives at 9 AM",)),
    _c("first we meet John, then we meet Sarah",
      "First we meet John, then we meet Sarah.", "basic", (), pres=True,
      preserved=("meet John, then we meet Sarah",)),
    _c("the shift runs from Monday to Friday",
      "The shift runs from Monday to Friday.", "basic", (), pres=True,
      preserved=("from Monday to Friday",)),
    # long + formatting / messages (12)
    _c("uh can you write a message to the team saying the the deployment is moved from Tuesday to Thursday and make it professional",
      "Dear team, the deployment has been moved from Tuesday to Thursday.",
      "mixed", ("filler", "repetition", "formatting"),
      fmt={"type": "polite_message", "items": ["the deployment is moved from Tuesday to Thursday"]},
      domain="email"),
    _c("make a list of everything we need for the trip like passport tickets charger and sunscreen",
      "- Passport\n- Tickets\n- Charger\n- Sunscreen", "mixed", ("filler", "formatting"),
      fmt={"type": "bullet_list", "items": ["passport", "tickets", "charger", "sunscreen"]},
      domain="lists"),
    _c("um write the agenda for Monday as a list first the budget then hiring then the roadmap",
      "- Budget\n- Hiring\n- Roadmap", "mixed", ("filler", "formatting"),
      fmt={"type": "bullet_list", "items": ["budget", "hiring", "roadmap"]}, domain="lists"),
    _c("so uh message Omar that I I can't make the meeting tomorrow and make it sound friendly",
      "Hi Omar, I can't make the meeting tomorrow.",
      "mixed", ("filler", "repetition", "formatting"),
      fmt={"type": "polite_message", "items": ["I can't make the meeting tomorrow"]},
      domain="messaging"),
    _c("make this a heading: the Monday standup notes",
      "# The Monday standup notes", "formatting", ("formatting",),
      fmt={"type": "heading", "items": ["the Monday standup notes"]}, domain="notes"),
    _c("put the action items into bullet points update the docs review the PR and deploy on Friday",
      "- Update the docs\n- Review the PR\n- Deploy on Friday", "mixed", ("filler", "formatting"),
      fmt={"type": "bullet_list", "items": ["update the docs", "review the PR", "deploy on Friday"]},
      domain="lists"),
    _c("can you write a message to the client saying the the report will be ready on Wednesday instead of Tuesday and make it professional",
      "Dear client, the report will be ready on Wednesday instead of Tuesday.",
      "mixed", ("repetition", "formatting"),
      fmt={"type": "polite_message", "items": ["the report will be ready on Wednesday instead of Tuesday"]},
      domain="email"),
    _c("um make a heading for the notes about the Friday client call",
      "# The Friday client call", "mixed", ("filler", "formatting"),
      fmt={"type": "heading", "items": ["the Friday client call"]}, domain="notes"),
    _c("make it polite: The payment is late.",
      "Could you please take care of the payment, which is late?", "formatting", ("formatting",),
      fmt={"type": "polite_message", "items": ["The payment is late."]}, domain="email"),
    _c("uh so write the shopping list milk eggs and paratha",
      "- Milk\n- Eggs\n- Paratha", "mixed", ("filler", "formatting"),
      fmt={"type": "bullet_list", "items": ["milk", "eggs", "paratha"]}, domain="lists"),
    _c("make this sound professional: hey the server is broken",
      "Hello, the server is currently experiencing an outage.", "formatting", ("formatting",),
      fmt={"type": "polite_message", "items": ["the server is broken"]}, domain="technical"),
    _c("um can you make a list of the rooms we need to book the conference room the lab and the studio",
      "- The conference room\n- The lab\n- The studio", "mixed", ("filler", "formatting"),
      fmt={"type": "bullet_list", "items": ["the conference room", "the lab", "the studio"]},
      domain="lists"),
    # long + reformulation (8)
    _c("okay so I was thinking maybe we could push the launch to next month—actually you know what let's just do it on Friday",
      "Let's just do it on Friday.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "okay so I was thinking maybe we could push the launch to next month", "kept": "Let's just do it on Friday."},)),
    _c("I wanted to say that maybe we could take the train—actually let's just drive to the office.",
      "Let's just drive to the office.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "I wanted to say that maybe we could take the train", "kept": "Let's just drive to the office."},)),
    _c("so the thing is I was going to bring up the budget first, but anyway, send the files to Priya today.",
      "Send the files to Priya today.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "so the thing is I was going to bring up the budget first", "kept": "Send the files to Priya today."},)),
    _c("what I meant to say was we should wait for the numbers, I mean, call the vendor now.",
      "Call the vendor now.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "what I meant to say was we should wait for the numbers", "kept": "Call the vendor now."},)),
    _c("we should probably, let me rephrase that, we should definitely book the tickets tonight.",
      "We should definitely book the tickets tonight.", "reformulation", ("reformulation",),
      reform=({"mode": "restart", "abandoned": "we should probably", "kept": "We should definitely book the tickets tonight."},)),
    _c("I'll handle the, I mean, I'll handle the presentation slides myself.",
      "I'll handle the presentation slides myself.", "reformulation", ("reformulation",),
      reform=({"mode": "restart", "abandoned": "I'll handle the", "kept": "I'll handle the presentation slides myself."},)),
    _c("hmm you know what maybe we should skip the demo and just, so actually, send the agenda and we'll talk on Monday.",
      "Send the agenda and we'll talk on Monday.", "reformulation", ("reformulation",),
      reform=({"mode": "distract", "abandoned": "hmm you know what maybe we should skip the demo and just", "kept": "Send the agenda and we'll talk on Monday."},)),
    _c("can we, wait, can we move the review to after lunch.",
      "Can we move the review to after lunch?", "reformulation", ("reformulation",),
      reform=({"mode": "restart", "abandoned": "can we", "kept": "Can we move the review to after lunch?"},)),
]

# --------------------------------------------------------------------------
# EXTREME — long + corrections + formatting + entities (50)
# --------------------------------------------------------------------------

EXTREME = [
    # spec §11 cases verbatim (6)
    _c("uh i i wanted to send sarah the project files tomorrow actually no wednesday morning",
      "I wanted to send Sarah the project files Wednesday morning.",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("tomorrow", "wednesday morning"),), domain="work",
      notes="spec §11 smoke test — Experiment 1 failed this"),
    _c("okay so basically I was thinking we could meet Tuesday no wait Thursday afternoon because John isn't available Tuesday",
      "I was thinking we could meet Thursday afternoon because John isn't available Tuesday.",
      "mixed", ("filler", "self_correction"),
      corr=(("Tuesday", "Thursday afternoon"),), domain="scheduling",
      notes="spec §11 — the second Tuesday is meaningful and must survive"),
    _c("uh can you write a message to Sarah saying I I won't be able to make it tomorrow no Wednesday morning and make it professional",
      "Hi Sarah, I won't be able to make it Wednesday morning.",
      "mixed", ("filler", "repetition", "self_correction", "formatting"),
      corr=(("tomorrow", "Wednesday morning"),),
      fmt={"type": "polite_message", "items": ["I won't be able to make it Wednesday morning"]},
      domain="messaging", notes="spec §11"),
    _c("make a list of milk eggs bread actually remove eggs and add paratha",
      "- Milk\n- Bread\n- Paratha", "mixed", ("self_correction", "formatting"),
      corr=(("eggs", "paratha"),),
      fmt={"type": "bullet_list", "items": ["milk", "bread", "paratha"]},
      domain="lists", notes="spec §11 — list edit"),
    _c("tell alex the the server crashed because of port 8080 wait port 3000 and make it sound professional",
      "Hi Alex, the server crashed due to port 3000.",
      "mixed", ("repetition", "self_correction", "formatting"),
      corr=(("8080", "3000"),),
      fmt={"type": "polite_message", "items": ["the server crashed due to port 3000"]},
      domain="technical", notes="spec §11"),
    _c("uh i i think we should meet tuesday no wait make that thursday afternoon at 4",
      "I think we should meet Thursday afternoon at 4.",
      "mixed", ("filler", "repetition", "self_correction"),
      corr=(("tuesday", "thursday afternoon"),), domain="scheduling",
      notes="spec §5 mixed example"),
    # formatting + correction + repetition (10)
    _c("um make a list of apples oranges and bananas wait remove bananas and add grapes",
      "- Apples\n- Oranges\n- Grapes", "mixed", ("filler", "self_correction", "formatting"),
      corr=(("bananas", "grapes"),),
      fmt={"type": "bullet_list", "items": ["apples", "oranges", "grapes"]}, domain="lists"),
    _c("uh put milk bread and eggs into a bulleted list no wait swap eggs for cheese",
      "- Milk\n- Bread\n- Cheese", "mixed", ("filler", "self_correction", "formatting"),
      corr=(("eggs", "cheese"),),
      fmt={"type": "bullet_list", "items": ["milk", "bread", "cheese"]}, domain="lists"),
    _c("make a heading for the the Tuesday review no sorry the Thursday review",
      "# The Thursday review", "mixed", ("repetition", "self_correction", "formatting"),
      corr=(("Tuesday", "Thursday"),),
      fmt={"type": "heading", "items": ["the Thursday review"]}, domain="notes"),
    _c("can you write a message to Rina saying I I'll be late today no tomorrow and make it polite",
      "Hi Rina, I'll be late tomorrow.",
      "mixed", ("repetition", "self_correction", "formatting"),
      corr=(("today", "tomorrow"),),
      fmt={"type": "polite_message", "items": ["I'll be late tomorrow"]}, domain="messaging"),
    _c("make this sound professional: hey can you fix the the dashboard it's been down since Monday no Sunday",
      "Hello, could you fix the dashboard? It has been down since Sunday.",
      "mixed", ("repetition", "self_correction", "formatting"),
      corr=(("Monday", "Sunday"),),
      fmt={"type": "polite_message", "items": ["fix the dashboard, it has been down since Sunday"]},
      domain="technical"),
    _c("uh make a list of the three priorities quality speed and cost actually replace cost with security",
      "- Quality\n- Speed\n- Security", "mixed", ("filler", "self_correction", "formatting"),
      corr=(("cost", "security"),),
      fmt={"type": "bullet_list", "items": ["quality", "speed", "security"]}, domain="lists"),
    _c("make the the heading about the Friday standup no wait the Monday standup",
      "# The Monday standup", "mixed", ("repetition", "self_correction", "formatting"),
      corr=(("Friday", "Monday"),),
      fmt={"type": "heading", "items": ["the Monday standup"]}, domain="notes"),
    _c("write a polite message to Tom saying the the package will arrive Wednesday no Thursday",
      "Hi Tom, the package will arrive Thursday.",
      "mixed", ("repetition", "self_correction", "formatting"),
      corr=(("Wednesday", "Thursday"),),
      fmt={"type": "polite_message", "items": ["the package will arrive Thursday"]},
      domain="messaging"),
    _c("uh put rice pasta and sauce in a list and then actually swap rice for noodles",
      "- Noodles\n- Pasta\n- Sauce", "mixed", ("filler", "self_correction", "formatting"),
      corr=(("rice", "noodles"),),
      fmt={"type": "bullet_list", "items": ["noodles", "pasta", "sauce"]}, domain="lists"),
    _c("make it professional: hey the the API is returning errors on port 8080 wait port 9090",
      "Hello, the API is currently returning errors on port 9090.",
      "mixed", ("repetition", "self_correction", "formatting"),
      corr=(("8080", "9090"),),
      fmt={"type": "polite_message", "items": ["the API is returning errors on port 9090"]},
      domain="technical"),
    # filler + reformulation (5) — combination NOT in the training matrix
    _c("um I wanted to say that maybe we could review the contract first—actually just send it to Lucas today.",
      "Just send it to Lucas today.", "reformulation", ("filler", "reformulation"),
      reform=({"mode": "distract", "abandoned": "I wanted to say that maybe we could review the contract first", "kept": "Just send it to Lucas today."},),
      domain="work", notes="unseen combination: filler+reformulation"),
    _c("uh so the thing is I was thinking about moving the call, anyway, keep it at 3 PM.",
      "Keep it at 3 PM.", "reformulation", ("filler", "reformulation"),
      reform=({"mode": "distract", "abandoned": "so the thing is I was thinking about moving the call", "kept": "Keep it at 3 PM."},),
      domain="scheduling", notes="unseen combination"),
    _c("um you know what let's not talk about the budget now, so actually, call the client at 4.",
      "Call the client at 4.", "reformulation", ("filler", "reformulation"),
      reform=({"mode": "distract", "abandoned": "you know what let's not talk about the budget now", "kept": "Call the client at 4."},),
      domain="work", notes="unseen combination"),
    _c("hmm I was going to say we should wait, but actually, ship it tonight.",
      "Ship it tonight.", "reformulation", ("filler", "reformulation"),
      reform=({"mode": "distract", "abandoned": "I was going to say we should wait", "kept": "Ship it tonight."},),
      domain="technical", notes="unseen combination"),
    _c("uh we should, I mean, we should confirm the venue by noon.",
      "We should confirm the venue by noon.", "reformulation", ("filler", "reformulation"),
      reform=({"mode": "restart", "abandoned": "we should", "kept": "We should confirm the venue by noon."},),
      domain="scheduling", notes="unseen combination"),
    # triple same-kind corrections (5) — the speaker changes their mind twice
    _c("let's meet Tuesday no Thursday no wait Monday",
      "Let's meet Monday.", "correction", ("self_correction", "self_correction", "self_correction"),
      corr=(("Tuesday", "Thursday"), ("Thursday", "Monday")), domain="scheduling",
      notes="triple correction on dates"),
    _c("the flight is at 6 PM no 7 PM actually make it 8 PM",
      "The flight is at 8 PM.", "correction", ("self_correction", "self_correction", "self_correction"),
      corr=(("6 PM", "7 PM"), ("7 PM", "8 PM")), domain="travel"),
    _c("send it to Priya no Omar wait no actually send it to Hana",
      "Send it to Hana.", "correction", ("self_correction", "self_correction", "self_correction"),
      corr=(("Priya", "Omar"), ("Omar", "Hana")), domain="messaging"),
    _c("we need version 2 no version 3 hold on version 4",
      "We need version 4.", "correction", ("self_correction", "self_correction", "self_correction"),
      corr=(("version 2", "version 3"), ("version 3", "version 4")), domain="technical"),
    _c("book the room in Berlin no Munich wait make that Vienna",
      "Book the room in Vienna.", "correction", ("self_correction", "self_correction", "self_correction"),
      corr=(("Berlin", "Munich"), ("Munich", "Vienna")), domain="travel"),
    # corrections inside long lists (5)
    _c("for the trip I need to pack the charger the headphones the passport and the camera no wait swap the camera for the tablet",
      "For the trip I need to pack the charger, the headphones, the passport, and the tablet.",
      "mixed", ("self_correction",), corr=(("the camera", "the tablet"),), domain="travel"),
    _c("the agenda covers the budget the hiring plan the roadmap and the offsite no sorry drop the offsite and add the security review",
      "The agenda covers the budget, the hiring plan, the roadmap, and the security review.",
      "mixed", ("self_correction",), corr=(("the offsite", "the security review"),), domain="work"),
    _c("we need to buy tomatoes onions garlic and ginger actually skip the garlic and add peppers",
      "We need to buy tomatoes, onions, ginger, and peppers.",
      "mixed", ("self_correction",), corr=(("garlic", "peppers"),), domain="shopping"),
    _c("the stops on the way are the bank the post office the pharmacy and the market no wait cut the market",
      "The stops on the way are the bank, the post office, and the pharmacy.",
      "mixed", ("self_correction",), corr=(("the market", ""),), domain="everyday",
      notes="deletion-style correction: a listed item is removed"),
    _c("invite Tariq Nadia Omar and Leila no actually drop Leila and add Samir",
      "Invite Tariq, Nadia, Omar, and Samir.",
      "mixed", ("self_correction",), corr=(("Leila", "Samir"),), domain="everyday"),
    # multi-entity stress (10)
    _c("uh I I need to book a flight from Dubai to Berlin on the 12th of March arriving before 6 PM and the hotel near the airport for three nights",
      "I need to book a flight from Dubai to Berlin on the 12th of March arriving before 6 PM, and the hotel near the airport for three nights.",
      "mixed", ("filler", "repetition"), domain="travel"),
    _c("so the client wants the report by Friday at 5 PM and they want Priya and Omar on the call and the price needs to stay under 9000 dollars",
      "The client wants the report by Friday at 5 PM and they want Priya and Omar on the call, and the price needs to stay under 9000 dollars.",
      "mixed", ("filler",), domain="work"),
    _c("um the deployment window is Sunday night at 11 and the rollback plan is Monday at 9 AM and Tom will be on call with the team in Berlin",
      "The deployment window is Sunday night at 11 and the rollback plan is Monday at 9 AM, and Tom will be on call with the team in Berlin.",
      "mixed", ("filler",), domain="technical"),
    _c("uh we're meeting Maria at the cafe on Wednesday at 2 PM and then driving to the warehouse to pick up the chairs before 5",
      "We're meeting Maria at the cafe on Wednesday at 2 PM, and then driving to the warehouse to pick up the chairs before 5.",
      "mixed", ("filler",), domain="everyday"),
    _c("um I'll drop the kids at school at 8 then go to the gym and after that I have lunch with Hana at the mall at 1",
      "I'll drop the kids at school at 8, then go to the gym, and after that I have lunch with Hana at the mall at 1.",
      "mixed", ("filler",), domain="everyday"),
    _c("uh the invoice for 1250 dollars goes to the client in Tokyo and the contract for 800 goes to the office in London and both are due on the 30th",
      "The invoice for 1250 dollars goes to the client in Tokyo, and the contract for 800 goes to the office in London, and both are due on the 30th.",
      "mixed", ("filler",), domain="work"),
    _c("so the train to Vienna leaves at 7:30 AM and the connection in Prague is at 11 and we arrive by 4 PM",
      "The train to Vienna leaves at 7:30 AM, and the connection in Prague is at 11, and we arrive by 4 PM.",
      "mixed", ("filler",), domain="travel"),
    _c("um the server in the lab runs on port 3000 and the backup runs at midnight and Oskar checks the logs every morning at 9",
      "The server in the lab runs on port 3000, and the backup runs at midnight, and Oskar checks the logs every morning at 9.",
      "mixed", ("filler",), domain="technical"),
    _c("uh I need milk eggs flour and sugar for the cake and the party is on Saturday at 6 at Amara's place",
      "I need milk, eggs, flour, and sugar for the cake, and the party is on Saturday at 6 at Amara's place.",
      "mixed", ("filler",), domain="shopping"),
    # long no-change adversarial (10) — long clean text, nothing to fix
    _c("The meeting will start at 3 PM on Tuesday in the conference room and everyone should bring their laptops.",
      "The meeting will start at 3 PM on Tuesday in the conference room and everyone should bring their laptops.",
      "basic", (), domain="work"),
    _c("I checked the numbers twice and the total comes to 480 dollars after the discount.",
      "I checked the numbers twice and the total comes to 480 dollars after the discount.",
      "basic", (), pres=True, preserved=("the numbers twice",)),
    _c("The train leaves at 8, stops in Paris for twenty minutes, and arrives in Berlin at 3 PM.",
      "The train leaves at 8, stops in Paris for twenty minutes, and arrives in Berlin at 3 PM.",
      "basic", (), domain="travel"),
    _c("Please send the updated contract to Maria and copy Omar before Friday at noon.",
      "Please send the updated contract to Maria and copy Omar before Friday at noon.",
      "basic", (), domain="work"),
    _c("The library closes at 9 PM on weekdays but stays open until 11 on Saturday.",
      "The library closes at 9 PM on weekdays but stays open until 11 on Saturday.",
      "basic", (), domain="everyday"),
    _c("We compared the two vendors and the second one is cheaper by 15 percent.",
      "We compared the two vendors and the second one is cheaper by 15 percent.",
      "basic", (), domain="work"),
    _c("The recipe calls for 200 grams of flour, three eggs, and a cup of milk.",
      "The recipe calls for 200 grams of flour, three eggs, and a cup of milk.",
      "basic", (), domain="shopping"),
    _c("The server restarts every night at midnight and the backups run right after.",
      "The server restarts every night at midnight and the backups run right after.",
      "basic", (), domain="technical"),
    _c("Our flight lands in Lisbon at 6 PM and the shuttle to the hotel leaves every half hour.",
      "Our flight lands in Lisbon at 6 PM and the shuttle to the hotel leaves every half hour.",
      "basic", (), domain="travel"),
    _c("The exam covers chapters 4 through 7 and the review session is on Wednesday at 5 PM.",
      "The exam covers chapters 4 through 7 and the review session is on Wednesday at 5 PM.",
      "basic", (), domain="university"),
]

ALL_LEVELS = {"easy": EASY, "medium": MEDIUM, "hard": HARD, "extreme": EXTREME}


# --------------------------------------------------------------------------
# Finalizer + validator
# --------------------------------------------------------------------------

def _finalize(entry: dict, level: str, idx: int) -> dict:
    """Turn a compact authored entry into a full record (entities derived from
    the clean output; correction entries get kind + marker filled in)."""
    corrections = []
    for corr in entry["corr"]:
        abandoned, kept = corr[0], corr[1]
        kind = None
        if kept:  # deletion-style corrections (kept == "") have no kind
            spans = entity_spans(kept)
            kind = spans[0].kind if spans else None
        marker = corr[2] if len(corr) > 2 else ""
        if not marker and kept and abandoned:
            # derive marker by scanning the input between abandoned and kept
            inp_low = entry["in"].lower()
            a, k = abandoned.lower(), kept.lower()
            pos_a = inp_low.find(a)
            pos_k = inp_low.find(k, pos_a + len(a)) if pos_a >= 0 else -1
            if pos_a >= 0 and pos_k > pos_a:
                marker = entry["in"][pos_a + len(abandoned):pos_k].strip(" ,-")
        corrections.append({"abandoned": abandoned, "kept": kept, "kind": kind, "marker": marker})

    transforms = list(entry["tr"])
    return {
        "id": f"stress_{idx:04d}",
        "level": level,
        "input": entry["in"],
        "output": entry["out"],
        "category": entry["cat"],
        "transformations": transforms,
        "num_transformations": len(transforms),
        "difficulty": {"easy": "easy", "medium": "medium", "hard": "hard", "extreme": "hard"}[level],
        "domain": entry["domain"],
        "entities": entities_from_text(entry["out"]),
        "corrections": corrections,
        "reformulations": [dict(r) for r in entry["reform"]],
        "formatting": dict(entry["fmt"]) if entry["fmt"] else None,
        "preservation_case": entry["pres"],
        "preserved": list(entry["preserved"]),
        "combo": None,
        "template_id": None,
        "seed": int(hashlib.md5(f"stress_{idx:04d}".encode()).hexdigest()[:8], 16),
        "generator_version": "handwritten",
        "notes": entry["notes"],
    }


def build_stress_records() -> list[dict]:
    records = []
    idx = 1
    for level in LEVELS:
        entries = ALL_LEVELS[level]
        if len(entries) != EXPECTED_PER_LEVEL:
            raise StressError(f"{level} has {len(entries)} entries, expected {EXPECTED_PER_LEVEL}")
        for entry in entries:
            records.append(_finalize(entry, level, idx))
            idx += 1
    if len(records) != 200:
        raise StressError(f"total {len(records)} records, expected 200")
    return records


def validate_stress(records: list[dict]) -> list[str]:
    """Schema-level validation; returns a list of problems (empty = OK)."""
    problems = []
    counts = {level: 0 for level in LEVELS}
    seen_ids = set()
    for r in records:
        if not r["id"].startswith("stress_"):
            problems.append(f"bad id {r['id']}")
        if r["id"] in seen_ids:
            problems.append(f"duplicate id {r['id']}")
        seen_ids.add(r["id"])
        if r["level"] not in LEVELS:
            problems.append(f"bad level {r['level']!r} in {r['id']}")
        counts[r["level"]] = counts.get(r["level"], 0) + 1
        if not r["input"].strip() or not r["output"].strip():
            problems.append(f"empty input/output in {r['id']}")
        if entities_from_text(r["output"]) != r["entities"]:
            problems.append(f"entity metadata mismatch in {r['id']}")
        corrs = r["corrections"]
        for i, c in enumerate(corrs):
            # Chained corrections (next abandoned == this kept, e.g. "Tuesday no
            # Thursday no Monday"): the intermediate kept value is superseded
            # and must NOT survive. Independent corrections: every kept value
            # must survive.
            chained = (
                i < len(corrs) - 1
                and corrs[i + 1]["abandoned"].lower() == c["kept"].lower()
            )
            if c["kept"] and not chained:
                if c["kept"].lower() not in r["output"].lower():
                    problems.append(f"kept {c['kept']!r} missing from output in {r['id']}")
            elif c["kept"] and chained and c["kept"].lower() in r["output"].lower():
                problems.append(f"superseded kept {c['kept']!r} survived in output in {r['id']}")
            if c["abandoned"] and c["abandoned"].lower() not in r["input"].lower():
                problems.append(f"abandoned {c['abandoned']!r} missing from input in {r['id']}")
        for sub in r["preserved"]:
            if sub not in r["output"]:
                problems.append(f"preserved {sub!r} missing from output in {r['id']}")
    for level in LEVELS:
        if counts[level] != EXPECTED_PER_LEVEL:
            problems.append(f"{level} has {counts[level]} records, expected {EXPECTED_PER_LEVEL}")
    return problems


def write_stress_jsonl(path: str | Path) -> int:
    records = build_stress_records()
    write_jsonl(path, records)
    return len(records)


if __name__ == "__main__":
    records = build_stress_records()
    problems = validate_stress(records)
    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(records)} records, {len(problems)} problems")
