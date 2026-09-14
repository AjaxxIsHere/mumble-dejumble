"""V4 Rambler Stress Test — 50 HAND-WRITTEN records for V4 validation.

These complement the existing 200-record stress test (data/stress_test.py)
with V4-specific patterns:
- Nyra-style conversational disfluencies (realistic speech)
- Augmentation pipeline edge cases
- Multi-token corrections with dates/times/names
- False starts and clause restarts
- Long multi-clause speech with corrections
- Formatting + correction combinations
- No-op long text
- Adversarial preservation cases

Ids use `stress_v4_` prefix (disjoint from existing `stress_` prefix).
"""

from pathlib import Path

from data.common import write_jsonl
from data.generators.composer import entities_from_text
from data.generators.entities import entity_spans

LEVELS = ("easy", "medium", "hard", "extreme")
EXPECTED_PER_LEVEL_V4 = 50


def _c(in_, out, cat, tr=(), corr=(), reform=(), fmt=None, pres=False,
       preserved=(), domain="everyday", notes=None):
    return {
        "in": in_, "out": out, "cat": cat, "tr": tuple(tr),
        "corr": tuple(corr), "reform": tuple(reform), "fmt": fmt,
        "pres": pres, "preserved": tuple(preserved), "domain": domain, "notes": notes,
    }


# --------------------------------------------------------------------------
# EASY — simple Nyra-style cleanup (50)
# --------------------------------------------------------------------------
EASY_V4 = [
    # Nyra-style conversational fillers
    _c("so I I kind of gave up on the idea of using Quicken at least for now",
       "I gave up on the idea of using Quicken, at least for now.", "filler", ("filler", "repetition"),
       domain="everyday", notes="Nyra-style conversation"),
    _c("uh well I actually my dad's almost ninety and he lives by himself",
       "Well, my dad's almost ninety and he lives by himself.", "filler", ("filler",),
       domain="everyday"),
    _c("um you know I think it's great I really and I look for more of that",
       "I think it's great, I really, and I look for more of that.", "filler", ("filler",),
       domain="everyday"),
    _c("basically um we're we're out of time for the meeting",
       "We're out of time for the meeting.", "mixed", ("filler", "repetition"),
       domain="work"),
    _c("so yeah the report is done and I sent it to everyone",
       "The report is done and I sent it to everyone.", "filler", ("filler",),
       domain="work"),

    # Simple word repetitions
    _c("I I need to finish this today before the deadline",
       "I need to finish this today before the deadline.", "repetition", ("repetition",),
       domain="work"),
    _c("the the meeting is tomorrow at three",
       "The meeting is tomorrow at three.", "repetition", ("repetition",),
       domain="scheduling"),
    _c("can can you send me the files",
       "Can you send me the files?", "repetition", ("repetition",),
       domain="messaging"),
    _c("it's it's not working properly",
       "It's not working properly.", "repetition", ("repetition",),
       domain="technical"),
    _c("we we should probably order more supplies",
       "We should probably order more supplies.", "repetition", ("repetition",),
       domain="shopping"),

    # Single date corrections
    _c("the meeting is on Tuesday no wait Thursday",
       "The meeting is on Thursday.", "correction", ("self_correction",),
       corr=(("Tuesday", "Thursday"),), domain="scheduling"),
    _c("I'll call you tomorrow actually tonight",
       "I'll call you tonight.", "correction", ("self_correction",),
       corr=(("tomorrow", "tonight"),), domain="messaging"),
    _c("the deadline is Friday no Monday",
       "The deadline is Monday.", "correction", ("self_correction",),
       corr=(("Friday", "Monday"),), domain="work"),
    _c("let's meet next Wednesday actually next Thursday",
       "Let's meet next Thursday.", "correction", ("self_correction",),
       corr=(("next Wednesday", "next Thursday"),), domain="scheduling"),

    # Single name corrections
    _c("send it to John actually Sarah",
       "Send it to Sarah.", "correction", ("self_correction",),
       corr=(("John", "Sarah"),), domain="messaging"),
    _c("ask Priya no Aisha to review the document",
       "Ask Aisha to review the document.", "correction", ("self_correction",),
       corr=(("Priya", "Aisha"),), domain="work"),
    _c("tell Omar about the change actually tell Nadia",
       "Tell Nadia about the change.", "correction", ("self_correction",),
       corr=(("Omar", "Nadia"),), domain="messaging"),

    # Single time corrections
    _c("the call is at 3 PM no wait 4:30 PM",
       "The call is at 4:30 PM.", "correction", ("self_correction",),
       corr=(("3 PM", "4:30 PM"),), domain="scheduling"),
    _c("let's meet at noon actually at 2 PM",
       "Let's meet at 2 PM.", "correction", ("self_correction",),
       corr=(("noon", "2 PM"),), domain="scheduling"),

    # Single number corrections
    _c("I need two copies no three copies",
       "I need three copies.", "correction", ("self_correction",),
       corr=(("two", "three"),), domain="work"),
    _c("the price is fifty no forty-five dollars",
       "The price is forty-five dollars.", "correction", ("self_correction",),
       corr=(("fifty", "forty-five"),), domain="shopping"),

    # Technical entity corrections
    _c("deploy to port 8080 actually port 3000",
       "Deploy to port 3000.", "correction", ("self_correction",),
       corr=(("8080", "3000"),), domain="technical"),
    _c("use the MySQL database no Postgres",
       "Use the Postgres database.", "correction", ("self_correction",),
       corr=(("MySQL", "Postgres"),), domain="technical"),

    # No-op (already clean)
    _c("The meeting is at 3 PM.", "The meeting is at 3 PM.", "basic", (),
       domain="scheduling"),
    _c("Send the report to Sarah by Friday.", "Send the report to Sarah by Friday.", "basic", (),
       domain="work"),
    _c("I need to buy milk, eggs, and bread.", "I need to buy milk, eggs, and bread.", "basic", (),
       domain="shopping"),
    _c("The server is running on port 3000.", "The server is running on port 3000.", "basic", (),
       domain="technical"),
    _c("Call the client back about the invoice.", "Call the client back about the invoice.", "basic", (),
       domain="work"),

    # Punctuation/capitalization only
    _c("is the report ready", "Is the report ready?", "basic", (),
       domain="work"),
    _c("the meeting is on tuesday in dubai", "The meeting is on Tuesday in Dubai.", "basic", (),
       domain="scheduling"),
    _c("send the files to sarah on friday", "Send the files to Sarah on Friday.", "basic", (),
       domain="messaging"),
    _c("i need 2 copies of the contract", "I need 2 copies of the contract.", "basic", (),
       domain="work"),
    _c("dinner at 7 tonight", "Dinner at 7 tonight.", "basic", (),
       domain="everyday"),

    # Filler + single correction (easy composition)
    _c("uh the meeting is on Tuesday no wait Thursday",
       "The meeting is on Thursday.", "mixed", ("filler", "self_correction"),
       corr=(("Tuesday", "Thursday"),), domain="scheduling"),
    _c("um send it to John actually Sarah",
       "Send it to Sarah.", "mixed", ("filler", "self_correction"),
       corr=(("John", "Sarah"),), domain="messaging"),
    _c("so like the price is 50 no 45 dollars",
       "The price is 45 dollars.", "mixed", ("filler", "self_correction"),
       corr=(("50", "45"),), domain="shopping"),

    # Repetition + single correction
    _c("I I need the report by Thursday no Friday",
       "I need the report by Friday.", "mixed", ("repetition", "self_correction"),
       corr=(("Thursday", "Friday"),), domain="work"),
    _c("the the meeting is at 2 PM no wait 3 PM",
       "The meeting is at 3 PM.", "mixed", ("repetition", "self_correction"),
       corr=(("2 PM", "3 PM"),), domain="scheduling"),
    _c("send send the files to the office no the lab",
       "Send the files to the lab.", "mixed", ("repetition", "self_correction"),
       corr=(("the office", "the lab"),), domain="work"),

    # Simple formatting
    _c("make a list of milk eggs and bread",
       "- Milk\n- Eggs\n- Bread", "formatting", ("formatting",),
       fmt={"type": "bullet_list", "items": ["milk", "eggs", "bread"]},
       domain="lists"),
    _c("make this a heading: notes from the Tuesday call",
       "# Notes from the Tuesday call", "formatting", ("formatting",),
       fmt={"type": "heading", "items": ["notes from the Tuesday call"]},
       domain="notes"),
    _c("make it polite: Send me the report by Friday.",
       "Could you please send me the report by Friday?", "formatting", ("formatting",),
       fmt={"type": "polite_message", "items": ["Send me the report by Friday."]},
       domain="email"),

    # Preservation cases
    _c("the meeting runs from 3 PM to 4 PM",
       "The meeting runs from 3 PM to 4 PM.", "basic", (), pres=True,
       preserved=("from 3 PM to 4 PM",), domain="scheduling"),
    _c("I was going to say Tuesday but Wednesday works better",
       "I was going to say Tuesday, but Wednesday works better.", "basic", (), pres=True,
       preserved=("I was going to say Tuesday, but Wednesday works better",),
       domain="scheduling", notes="adversarial: both dates meaningful"),

    # Long no-op
    _c("Please send the quarterly report to the entire team before the board meeting on Friday and make sure to include the budget analysis from last quarter",
       "Please send the quarterly report to the entire team before the board meeting on Friday and make sure to include the budget analysis from last quarter.",
       "basic", (), domain="work"),

    # Location corrections
    _c("let's meet at the office no the cafe",
       "Let's meet at the cafe.", "correction", ("self_correction",),
       corr=(("the office", "the cafe"),), domain="scheduling"),

    # Item corrections
    _c("buy the blue one no the green one",
       "Buy the green one.", "correction", ("self_correction",),
       corr=(("the blue one", "the green one"),), domain="shopping"),

    # Tech corrections
    _c("use React actually Vue for the frontend",
       "Use Vue for the frontend.", "correction", ("self_correction",),
       corr=(("React", "Vue"),), domain="technical"),

    # Filler + no-op (should still be clean)
    _c("um the meeting is at 3 PM",
       "The meeting is at 3 PM.", "filler", ("filler",),
       domain="scheduling"),

    # Repetition + no-op
    _c("I I think the report is ready",
       "I think the report is ready.", "repetition", ("repetition",),
       domain="work"),
]


# --------------------------------------------------------------------------
# MEDIUM — multiple disfluencies (50)
# --------------------------------------------------------------------------
MEDIUM_V4 = [
    # Filler + repetition (8)
    _c("um I I wanted to ask about the the budget for next quarter",
       "I wanted to ask about the budget for next quarter.", "mixed",
       ("filler", "repetition"), domain="work"),
    _c("so uh we we should reschedule the the call with the client",
       "We should reschedule the call with the client.", "mixed",
       ("filler", "repetition"), domain="scheduling"),
    _c("you know I I can't find the the keys to the server room",
       "I can't find the keys to the server room.", "mixed",
       ("filler", "repetition"), domain="technical"),
    _c("like the the store was was closed when I got there",
       "The store was closed when I got there.", "mixed",
       ("filler", "repetition"), domain="shopping"),
    _c("basically um we're we're out of time for the meeting today",
       "We're out of time for the meeting today.", "mixed",
       ("filler", "repetition"), domain="work"),
    _c("er I I need to to finish this report by tomorrow",
       "I need to finish this report by tomorrow.", "mixed",
       ("filler", "repetition"), domain="work"),
    _c("hmm the the wifi keeps keeps dropping in the conference room",
       "The wifi keeps dropping in the conference room.", "mixed",
       ("filler", "repetition"), domain="technical"),
    _c("well um he he said it's it's fine for now",
       "He said it's fine for now.", "mixed",
       ("filler", "repetition"), domain="everyday"),

    # Filler + correction (8)
    _c("uh can you send it to John actually no Sarah",
       "Can you send it to Sarah?", "mixed",
       ("filler", "self_correction"), corr=(("John", "Sarah"),), domain="messaging"),
    _c("so like meet me at 3 wait 4:30",
       "Meet me at 4:30.", "mixed",
       ("filler", "self_correction"), corr=(("3", "4:30"),), domain="scheduling"),
    _c("um the deadline is Friday I mean Monday",
       "The deadline is Monday.", "mixed",
       ("filler", "self_correction"), corr=(("Friday", "Monday"),), domain="work"),
    _c("basically we're going to Dubai no Abu Dhabi",
       "We're going to Abu Dhabi.", "mixed",
       ("filler", "self_correction"), corr=(("Dubai", "Abu Dhabi"),), domain="travel"),
    _c("you know the price is 50 no 45 dollars",
       "The price is 45 dollars.", "mixed",
       ("filler", "self_correction"), corr=(("50", "45"),), domain="shopping"),
    _c("err port 8080 sorry port 9090",
       "Port 9090.", "mixed",
       ("filler", "self_correction"), corr=(("8080", "9090"),), domain="technical"),
    _c("well I'll come Tuesday actually Wednesday",
       "I'll come Wednesday.", "mixed",
       ("filler", "self_correction"), corr=(("Tuesday", "Wednesday"),), domain="scheduling"),
    _c("ah give me two no three copies of the document",
       "Give me three copies of the document.", "mixed",
       ("filler", "self_correction"), corr=(("two", "three"),), domain="work"),

    # Repetition + correction (8)
    _c("I I need the the report by Thursday no Friday",
       "I need the report by Friday.", "mixed",
       ("repetition", "self_correction"), corr=(("Thursday", "Friday"),), domain="work"),
    _c("the the meeting is is at 2 PM no wait 3 PM",
       "The meeting is at 3 PM.", "mixed",
       ("repetition", "self_correction"), corr=(("2 PM", "3 PM"),), domain="scheduling"),
    _c("call call Priya no wait Aisha tomorrow morning",
       "Call Aisha tomorrow morning.", "mixed",
       ("repetition", "self_correction"), corr=(("Priya", "Aisha"),), domain="messaging"),
    _c("send send the files to to the office no the lab",
       "Send the files to the lab.", "mixed",
       ("repetition", "self_correction"), corr=(("the office", "the lab"),), domain="work"),
    _c("we we fly on on Monday actually Tuesday",
       "We fly on Tuesday.", "mixed",
       ("repetition", "self_correction"), corr=(("Monday", "Tuesday"),), domain="travel"),
    _c("it it costs 20 no 25 dollars for the subscription",
       "It costs 25 dollars for the subscription.", "mixed",
       ("repetition", "self_correction"), corr=(("20", "25"),), domain="shopping"),
    _c("the the server runs on on port 8080 wait port 3000",
       "The server runs on port 3000.", "mixed",
       ("repetition", "self_correction"), corr=(("8080", "3000"),), domain="technical"),
    _c("I'm I'm meeting Omar no no Hana at 5 PM",
       "I'm meeting Hana at 5 PM.", "mixed",
       ("repetition", "self_correction"), corr=(("Omar", "Hana"),), domain="scheduling"),

    # Multiple corrections (6)
    _c("the meeting is on Tuesday no Wednesday at 2 PM no 3 PM",
       "The meeting is on Wednesday at 3 PM.", "mixed",
       ("self_correction", "self_correction"),
       corr=(("Tuesday", "Wednesday"), ("2 PM", "3 PM")), domain="scheduling"),
    _c("send it to John no Sarah by Friday no Monday",
       "Send it to Sarah by Monday.", "mixed",
       ("self_correction", "self_correction"),
       corr=(("John", "Sarah"), ("Friday", "Monday")), domain="messaging"),
    _c("the server is on port 8080 no 3000 and the version is 2 no 3",
       "The server is on port 3000 and the version is 3.", "mixed",
       ("self_correction", "self_correction"),
       corr=(("8080", "3000"), ("2", "3")), domain="technical"),
    _c("we're flying to Dubai no Abu Dhabi on Monday no Tuesday",
       "We're flying to Abu Dhabi on Tuesday.", "mixed",
       ("self_correction", "self_correction"),
       corr=(("Dubai", "Abu Dhabi"), ("Monday", "Tuesday")), domain="travel"),
    _c("the price is 50 no 45 dollars and the quantity is 2 no 3",
       "The price is 45 dollars and the quantity is 3.", "mixed",
       ("self_correction", "self_correction"),
       corr=(("50", "45"), ("2", "3")), domain="shopping"),
    _c("the meeting is at 3 PM no 4 PM in the lab no the conference room",
       "The meeting is at 4 PM in the conference room.", "mixed",
       ("self_correction", "self_correction"),
       corr=(("3 PM", "4 PM"), ("the lab", "the conference room")), domain="work"),

    # Reformulation (6)
    _c("I wanted to say that maybe we could go over the numbers actually let's just meet tomorrow",
       "Let's just meet tomorrow.", "reformulation", ("reformulation",),
       reform=({"mode": "distract", "abandoned": "I wanted to say that maybe we could go over the numbers",
                "kept": "Let's just meet tomorrow."},)),
    _c("what I meant to say was we should maybe review the slides anyway we should ship on Friday",
       "We should ship on Friday.", "reformulation", ("reformulation",),
       reform=({"mode": "distract", "abandoned": "what I meant to say was we should maybe review the slides",
                "kept": "We should ship on Friday."},)),
    _c("I'll send the wait I'll send the file to Sarah",
       "I'll send the file to Sarah.", "reformulation", ("reformulation",),
       reform=({"mode": "restart", "abandoned": "I'll send the",
                "kept": "I'll send the file to Sarah."},)),
    _c("we should I mean we should book the tickets tonight",
       "We should book the tickets tonight.", "reformulation", ("reformulation",),
       reform=({"mode": "restart", "abandoned": "we should",
                "kept": "We should book the tickets tonight."},)),
    _c("hmm let me rephrase that the meeting is on hold",
       "The meeting is on hold.", "reformulation", ("reformulation",),
       reform=({"mode": "distract", "abandoned": "hmm let me rephrase that",
                "kept": "The meeting is on hold."},)),
    _c("can you what I meant is can you bring the charger",
       "Can you bring the charger?", "reformulation", ("reformulation",),
       reform=({"mode": "restart", "abandoned": "can you",
                "kept": "Can you bring the charger?"},)),

    # Preservation with noise (6)
    _c("uh the meeting runs from 3 PM to 4 PM",
       "The meeting runs from 3 PM to 4 PM.", "repetition", ("filler",),
       pres=True, preserved=("from 3 PM to 4 PM",), domain="scheduling"),
    _c("um we will review the report between Monday and Friday",
       "We will review the report between Monday and Friday.", "repetition", ("filler",),
       pres=True, preserved=("between Monday and Friday",), domain="work"),
    _c("the gate code is 4829 don't change it",
       "The gate code is 4829, don't change it.", "repetition", ("filler",),
       pres=True, preserved=("The gate code is 4829",), domain="technical"),
    _c("call me before 5 PM or after 7 PM",
       "Call me before 5 PM or after 7 PM.", "repetition", ("filler",),
       pres=True, preserved=("before 5 PM or after 7 PM",), domain="messaging"),
    _c("we compared version 2 and version 3",
       "We compared version 2 and version 3.", "basic", (), pres=True,
       preserved=("version 2 and version 3",), domain="technical"),
    _c("the shift runs from Monday to Friday",
       "The shift runs from Monday to Friday.", "basic", (), pres=True,
       preserved=("from Monday to Friday",), domain="work"),

    # Formatting (8)
    _c("put apples bananas and oranges in a bulleted list",
       "- Apples\n- Bananas\n- Oranges", "formatting", ("formatting",),
       fmt={"type": "bullet_list", "items": ["apples", "bananas", "oranges"]},
       domain="lists"),
    _c("turn this into a title: trip to Lisbon",
       "# Trip to Lisbon", "formatting", ("formatting",),
       fmt={"type": "heading", "items": ["trip to Lisbon"]}, domain="travel"),
    _c("write a polite version of Call Priya back about the invoice.",
       "I'd appreciate it if you could call Priya back about the invoice.",
       "formatting", ("formatting",),
       fmt={"type": "polite_message", "items": ["Call Priya back about the invoice."]},
       domain="email"),
    _c("make this a heading: the Monday standup notes",
       "# The Monday standup notes", "formatting", ("formatting",),
       fmt={"type": "heading", "items": ["the Monday standup notes"]}, domain="notes"),
    _c("put the action items into bullet points update the docs review the PR and deploy on Friday",
       "- Update the docs\n- Review the PR\n- Deploy on Friday", "formatting", ("formatting",),
       fmt={"type": "bullet_list", "items": ["update the docs", "review the PR", "deploy on Friday"]},
       domain="lists"),
    _c("can you make this sound polite: Book the conference room for Monday.",
       "When you get a chance, could you book the conference room for Monday?",
       "formatting", ("formatting",),
       fmt={"type": "polite_message", "items": ["Book the conference room for Monday."]},
       domain="work"),

    # Filler + formatting (2)
    _c("um make a list of milk eggs and paratha",
       "- Milk\n- Eggs\n- Paratha", "mixed", ("filler", "formatting"),
       fmt={"type": "bullet_list", "items": ["milk", "eggs", "paratha"]},
       domain="lists"),
    _c("uh so write the shopping list milk eggs and bread",
       "- Milk\n- Eggs\n- Bread", "mixed", ("filler", "formatting"),
       fmt={"type": "bullet_list", "items": ["milk", "eggs", "bread"]},
       domain="lists"),
]


# --------------------------------------------------------------------------
# HARD — long speech + multiple corrections (50)
# --------------------------------------------------------------------------
HARD_V4 = [
    # Long + 2 corrections (12)
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

    # Long + filler + repetition + correction (12)
    _c("uh basically I was talking to John yesterday and he said that we could probably have the meeting on Tuesday morning but actually I just remembered that he's travelling Tuesday so can you change it to Wednesday afternoon instead",
       "John said we could have the meeting Tuesday morning, but he's travelling Tuesday, so can you change it to Wednesday afternoon instead?",
       "mixed", ("filler", "repetition", "self_correction"),
       corr=(("Tuesday morning", "Wednesday afternoon"),), domain="scheduling",
       notes="spec §4 long example"),
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

    # Adversarial preservation (6)
    _c("I was going to say Tuesday, but Wednesday works better.",
       "I was going to say Tuesday, but Wednesday works better.", "basic", (), pres=True,
       preserved=("I was going to say Tuesday, but Wednesday works better",),
       notes="adversarial: both dates meaningful"),
    _c("the old price was 50 and the new price is 45",
       "The old price was 50 and the new price is 45.", "basic", (), pres=True,
       preserved=("The old price was 50",)),
    _c("the flight departs at 6 AM and arrives at 9 AM",
       "The flight departs at 6 AM and arrives at 9 AM.", "basic", (), pres=True,
       preserved=("departs at 6 AM and arrives at 9 AM",)),
    _c("first we meet John, then we meet Sarah",
       "First we meet John, then we meet Sarah.", "basic", (), pres=True,
       preserved=("meet John, then we meet Sarah",)),
    _c("we compared version 2 and version 3",
       "We compared version 2 and version 3.", "basic", (), pres=True,
       preserved=("version 2 and version 3",)),
    _c("the shift runs from Monday to Friday",
       "The shift runs from Monday to Friday.", "basic", (), pres=True,
       preserved=("from Monday to Friday",)),

    # Long + formatting (12)
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
       fmt={"type": "bullet_list", "items": ["budget", "hiring", "roadmap"]},
       domain="lists"),
    _c("so uh message Omar that I I can't make the meeting tomorrow and make it sound friendly",
       "Hi Omar, I can't make the meeting tomorrow.",
       "mixed", ("filler", "repetition", "formatting"),
       fmt={"type": "polite_message", "items": ["I can't make the meeting tomorrow"]},
       domain="messaging"),
    _c("can you write a message to the client saying the the report will be ready on Wednesday instead of Tuesday and make it professional",
       "Dear client, the report will be ready on Wednesday instead of Tuesday.",
       "mixed", ("repetition", "formatting"),
       fmt={"type": "polite_message", "items": ["the report will be ready on Wednesday instead of Tuesday"]},
       domain="email"),
    _c("um make a heading for the notes about the Friday client call",
       "# The Friday client call", "mixed", ("filler", "formatting"),
       fmt={"type": "heading", "items": ["the Friday client call"]},
       domain="notes"),
    _c("uh so write the shopping list milk eggs and paratha",
       "- Milk\n- Eggs\n- Paratha", "mixed", ("filler", "formatting"),
       fmt={"type": "bullet_list", "items": ["milk", "eggs", "paratha"]},
       domain="lists"),
    _c("make this sound professional: hey the server is broken",
       "Hello, the server is currently experiencing an outage.", "formatting", ("formatting",),
       fmt={"type": "polite_message", "items": ["the server is broken"]},
       domain="technical"),
    _c("make it polite: The payment is late.",
       "Could you please take care of the payment, which is late?", "formatting", ("formatting",),
       fmt={"type": "polite_message", "items": ["The payment is late."]},
       domain="email"),
    _c("can you make this sound polite: Book the conference room for Monday.",
       "When you get a chance, could you book the conference room for Monday?", "formatting", ("formatting",),
       fmt={"type": "polite_message", "items": ["Book the conference room for Monday."]},
       domain="work"),
    _c("turn this into a title: trip to Lisbon",
       "# Trip to Lisbon", "formatting", ("formatting",),
       fmt={"type": "heading", "items": ["trip to Lisbon"]}, domain="travel"),
    _c("make this a heading: the Kubernetes migration plan",
       "# The Kubernetes migration plan", "formatting", ("formatting",),
       fmt={"type": "heading", "items": ["the Kubernetes migration plan"]},
       domain="technical"),

    # Filler + repetition + correction (long, 6 more)
    _c("so uh the the database backup is is scheduled for tonight but but the the team wants it on the weekend instead",
       "The database backup is scheduled for tonight, but the team wants it on the weekend instead.",
       "mixed", ("filler", "repetition"), domain="technical"),
    _c("you know um I I was looking at the the schedule and I I realized the the trip overlaps with the the conference",
       "I was looking at the schedule and I realized the trip overlaps with the conference.",
       "mixed", ("filler", "repetition"), domain="travel"),
    _c("well um I I finally heard back from from the insurance company and and they they said the the claim is approved",
       "I finally heard back from the insurance company and they said the claim is approved.",
       "mixed", ("filler", "repetition"), domain="everyday"),
    _c("so I I think we should move the deadline from from Friday no wait Thursday because the team needs more time",
       "I think we should move the deadline from Thursday because the team needs more time.",
       "mixed", ("repetition", "self_correction"),
       corr=(("Friday", "Thursday"),), domain="work"),
    _c("um the the project is is behind schedule and we we need to hire two no three more developers",
       "The project is behind schedule and we need to hire three more developers.",
       "mixed", ("filler", "repetition", "self_correction"),
       corr=(("two", "three"),), domain="work"),
    _c("uh basically the the server is is overloaded and we we should scale from two no four instances",
       "The server is overloaded and we should scale from four instances.",
       "mixed", ("filler", "repetition", "self_correction"),
       corr=(("two", "four"),), domain="technical"),

    # Repetition + correction + formatting (2 more)
    _c("send send the update to the client no wait to the vendor and make it sound professional",
       "Dear vendor, the update has been sent.",
       "mixed", ("repetition", "self_correction", "formatting"),
       corr=(("the client", "the vendor"),),
       fmt={"type": "polite_message", "items": ["the update has been sent"]},
       domain="work"),
    _c("the the meeting is on on Tuesday no Wednesday and make it a heading",
       "# The Wednesday meeting", "mixed", ("repetition", "self_correction", "formatting"),
       corr=(("Tuesday", "Wednesday"),),
       fmt={"type": "heading", "items": ["The Wednesday meeting"]},
       domain="scheduling"),
]


# --------------------------------------------------------------------------
# EXTREME — long + corrections + formatting + entities (50)
# --------------------------------------------------------------------------
EXTREME_V4 = [
    # Filler + repetition + correction + formatting (10)
    _c("uh can you write a message to Sarah saying I I won't be able to make it tomorrow no Wednesday morning and make it professional",
       "Hi Sarah, I won't be able to make it Wednesday morning.",
       "mixed", ("filler", "repetition", "self_correction", "formatting"),
       corr=(("tomorrow", "Wednesday morning"),),
       fmt={"type": "polite_message", "items": ["I won't be able to make it Wednesday morning"]},
       domain="messaging", notes="spec §11"),
    _c("tell Alex the the server crashed because of port 8080 wait port 3000 and make it sound professional",
       "Hi Alex, the server crashed due to port 3000.",
       "mixed", ("repetition", "self_correction", "formatting"),
       corr=(("8080", "3000"),),
       fmt={"type": "polite_message", "items": ["the server crashed due to port 3000"]},
       domain="technical", notes="spec §11"),
    _c("make a list of milk eggs bread actually remove eggs and add paratha",
       "- Milk\n- Bread\n- Paratha", "mixed", ("self_correction", "formatting"),
       corr=(("eggs", "paratha"),),
       fmt={"type": "bullet_list", "items": ["milk", "bread", "paratha"]},
       domain="lists", notes="spec §11 — list edit"),
    _c("um make a list of apples oranges and bananas wait remove bananas and add grapes",
       "- Apples\n- Oranges\n- Grapes", "mixed", ("filler", "self_correction", "formatting"),
       corr=(("bananas", "grapes"),),
       fmt={"type": "bullet_list", "items": ["apples", "oranges", "grapes"]},
       domain="lists"),
    _c("uh put milk bread and eggs into a bulleted list no wait swap eggs for cheese",
       "- Milk\n- Bread\n- Cheese", "mixed", ("filler", "self_correction", "formatting"),
       corr=(("eggs", "cheese"),),
       fmt={"type": "bullet_list", "items": ["milk", "bread", "cheese"]},
       domain="lists"),
    _c("so message the team that the deadline is moved from Friday to Monday actually to Wednesday and make it professional",
       "Dear team, the deadline has been moved from Friday to Wednesday.",
       "mixed", ("filler", "self_correction", "formatting"),
       corr=(("Monday", "Wednesday"),),
       fmt={"type": "polite_message", "items": ["the deadline has been moved from Friday to Wednesday"]},
       domain="email"),
    _c("write the shopping list milk bread and eggs actually remove eggs and add paratha and make it a bulleted list",
       "- Milk\n- Bread\n- Paratha", "mixed", ("filler", "self_correction", "formatting"),
       corr=(("eggs", "paratha"),),
       fmt={"type": "bullet_list", "items": ["milk", "bread", "paratha"]},
       domain="lists"),
    _c("can you write a message to Omar saying the the meeting is moved to Thursday no wait Friday and make it polite",
       "Hi Omar, the meeting has been moved to Friday.",
       "mixed", ("filler", "repetition", "self_correction", "formatting"),
       corr=(("Thursday", "Friday"),),
       fmt={"type": "polite_message", "items": ["the meeting has been moved to Friday"]},
       domain="messaging"),
    _c("um make a list of the tasks we need to finish update the docs review the PR and deploy no wait remove deploy and add testing",
       "- Update the docs\n- Review the PR\n- Testing", "mixed", ("filler", "self_correction", "formatting"),
       corr=(("deploy", "testing"),),
       fmt={"type": "bullet_list", "items": ["update the docs", "review the PR", "testing"]},
       domain="work"),
    _c("write a polite message to the vendor saying the order is delayed actually the order is cancelled and make it professional",
       "Dear vendor, the order has been cancelled.",
       "mixed", ("filler", "self_correction", "formatting"),
       corr=(("delayed", "cancelled"),),
       fmt={"type": "polite_message", "items": ["the order is cancelled"]},
       domain="work"),

    # Long multi-clause + corrections (12)
    _c("okay so I was thinking maybe we could push the launch to next month actually you know what let's just do it on Friday",
       "Let's just do it on Friday.", "reformulation", ("reformulation",),
       reform=({"mode": "distract", "abandoned": "okay so I was thinking maybe we could push the launch to next month",
                "kept": "Let's just do it on Friday."},)),
    _c("I wanted to say that maybe we could take the train actually let's just drive to the office",
       "Let's just drive to the office.", "reformulation", ("reformulation",),
       reform=({"mode": "distract", "abandoned": "I wanted to say that maybe we could take the train",
                "kept": "Let's just drive to the office."},)),
    _c("so the thing is I was going to bring up the budget first but anyway send the files to Priya today",
       "Send the files to Priya today.", "reformulation", ("reformulation",),
       reform=({"mode": "distract", "abandoned": "so the thing is I was going to bring up the budget first",
                "kept": "Send the files to Priya today."},)),
    _c("we should probably let me rephrase that we should definitely book the tickets tonight",
       "We should definitely book the tickets tonight.", "reformulation", ("reformulation",),
       reform=({"mode": "restart", "abandoned": "we should probably",
                "kept": "We should definitely book the tickets tonight."},)),
    _c("I'll handle the I mean I'll handle the presentation slides myself",
       "I'll handle the presentation slides myself.", "reformulation", ("reformulation",),
       reform=({"mode": "restart", "abandoned": "I'll handle the",
                "kept": "I'll handle the presentation slides myself."},)),
    _c("can we wait can we move the review to after lunch",
       "Can we move the review to after lunch?", "reformulation", ("reformulation",),
       reform=({"mode": "restart", "abandoned": "can we",
                "kept": "Can we move the review to after lunch?"},)),
    _c("what I meant to say was we should wait for the numbers I mean call the vendor now",
       "Call the vendor now.", "reformulation", ("reformulation",),
       reform=({"mode": "distract", "abandoned": "what I meant to say was we should wait for the numbers",
                "kept": "Call the vendor now."},)),
    _c("hmm you know what maybe we should skip the demo and just so actually send the agenda and we'll talk on Monday",
       "Send the agenda and we'll talk on Monday.", "reformulation", ("reformulation",),
       reform=({"mode": "distract", "abandoned": "hmm you know what maybe we should skip the demo and just",
                "kept": "Send the agenda and we'll talk on Monday."},)),
    _c("uh i i wanted to send sarah the project files tomorrow actually no wednesday morning",
       "I wanted to send Sarah the project files Wednesday morning.",
       "mixed", ("filler", "repetition", "self_correction"),
       corr=(("tomorrow", "wednesday morning"),), domain="work",
       notes="spec §11 smoke test"),
    _c("okay so basically I was thinking we could meet Tuesday no wait Thursday afternoon because John isn't available Tuesday",
       "I was thinking we could meet Thursday afternoon because John isn't available Tuesday.",
       "mixed", ("filler", "self_correction"),
       corr=(("Tuesday", "Thursday afternoon"),), domain="scheduling",
       notes="spec §11 — second Tuesday is meaningful"),
    _c("uh i i think we should meet tuesday no wait make that thursday afternoon at 4",
       "I think we should meet Thursday afternoon at 4.",
       "mixed", ("filler", "repetition", "self_correction"),
       corr=(("tuesday", "thursday afternoon"),), domain="scheduling"),
    _c("so I need to call the vendor about the the shipment and also the the invoice needs to be updated from 500 no 450 dollars and we should schedule a follow-up meeting for next week",
       "I need to call the vendor about the shipment and also the invoice needs to be updated from 450 dollars and we should schedule a follow-up meeting for next week.",
       "mixed", ("filler", "repetition", "self_correction"),
       corr=(("500", "450"),), domain="work"),

    # Formatting + correction combinations (10)
    _c("make a list of apples oranges and bananas actually remove bananas and add grapes and pears",
       "- Apples\n- Oranges\n- Grapes\n- Pears", "mixed", ("self_correction", "formatting"),
       corr=(("bananas", "grapes"),),
       fmt={"type": "bullet_list", "items": ["apples", "oranges", "grapes", "pears"]},
       domain="lists"),
    _c("put milk bread eggs and cheese in a list wait remove the cheese and add yogurt",
       "- Milk\n- Bread\n- Eggs\n- Yogurt", "mixed", ("self_correction", "formatting"),
       corr=(("cheese", "yogurt"),),
       fmt={"type": "bullet_list", "items": ["milk", "bread", "eggs", "yogurt"]},
       domain="lists"),
    _c("write a message to the team about the meeting change from Monday no Friday and make it professional",
       "Dear team, the meeting has been changed from Monday to Friday.",
       "mixed", ("self_correction", "formatting"),
       corr=(("Monday", "Friday"),),
       fmt={"type": "polite_message", "items": ["the meeting has been changed from Monday to Friday"]},
       domain="email"),
    _c("make this a heading actually the Monday standup notes no the Tuesday standup notes",
       "# The Tuesday standup notes", "mixed", ("self_correction", "formatting"),
       corr=(("Monday", "Tuesday"),),
       fmt={"type": "heading", "items": ["the Tuesday standup notes"]},
       domain="notes"),
    _c("create a bullet list of the tasks update the docs actually no review the PR first then update the docs",
       "- Review the PR\n- Update the docs", "mixed", ("self_correction", "formatting"),
       corr=(("update the docs first", "review the PR first"),),
       fmt={"type": "bullet_list", "items": ["review the PR", "update the docs"]},
       domain="work"),
    _c("write a polite email to the vendor saying the shipment is delayed actually it's cancelled please",
       "Dear vendor, the shipment has been cancelled.",
       "mixed", ("self_correction", "formatting"),
       corr=(("delayed", "cancelled"),),
       fmt={"type": "polite_message", "items": ["the shipment is cancelled"]},
       domain="work"),
    _c("make a heading for the project plan no the project roadmap",
       "# The project roadmap", "mixed", ("self_correction", "formatting"),
       corr=(("plan", "roadmap"),),
       fmt={"type": "heading", "items": ["the project roadmap"]},
       domain="work"),
    _c("write a list of the action items deploy the update actually no ship the update review the code and test everything",
       "- Ship the update\n- Review the code\n- Test everything", "mixed", ("self_correction", "formatting"),
       corr=(("deploy", "ship"),),
       fmt={"type": "bullet_list", "items": ["ship the update", "review the code", "test everything"]},
       domain="technical"),
    _c("make it polite confirm the 4 PM meeting actually the 4:30 meeting",
       "Could you please confirm the 4:30 meeting?", "mixed", ("self_correction", "formatting"),
       corr=(("4 PM", "4:30"),),
       fmt={"type": "polite_message", "items": ["confirm the 4:30 meeting"]},
       domain="scheduling"),
    _c("can you write a heading for the budget review no the budget report",
       "# The budget report", "mixed", ("self_correction", "formatting"),
       corr=(("review", "report"),),
       fmt={"type": "heading", "items": ["the budget report"]},
       domain="work"),

    # Long no-op (8)
    _c("Please send the quarterly report to the entire team before the board meeting on Friday and make sure to include the budget analysis from last quarter",
       "Please send the quarterly report to the entire team before the board meeting on Friday and make sure to include the budget analysis from last quarter.",
       "basic", (), domain="work"),
    _c("The meeting is scheduled for next Tuesday at 3 PM in the conference room with the entire project team including the external consultants",
       "The meeting is scheduled for next Tuesday at 3 PM in the conference room with the entire project team including the external consultants.",
       "basic", (), domain="scheduling"),
    _c("I need to review the contract amendments before the deadline on Thursday and send my comments to the legal team",
       "I need to review the contract amendments before the deadline on Thursday and send my comments to the legal team.",
       "basic", (), domain="work"),
    _c("The server migration is planned for this weekend and we need to ensure all data is backed up before we start",
       "The server migration is planned for this weekend and we need to ensure all data is backed up before we start.",
       "basic", (), domain="technical"),
    _c("Please book the conference room for next Wednesday from 2 PM to 4 PM and invite the marketing team",
       "Please book the conference room for next Wednesday from 2 PM to 4 PM and invite the marketing team.",
       "basic", (), domain="scheduling"),
    _c("I want to schedule a call with the vendor to discuss the new pricing and delivery timeline for next quarter",
       "I want to schedule a call with the vendor to discuss the new pricing and delivery timeline for next quarter.",
       "basic", (), domain="work"),
    _c("The training session will cover the new security protocols and all team members must attend before the end of the month",
       "The training session will cover the new security protocols and all team members must attend before the end of the month.",
       "basic", (), domain="technical"),
    _c("We should review the project milestones from last sprint and plan the deliverables for the upcoming two-week cycle",
       "We should review the project milestones from last sprint and plan the deliverables for the upcoming two-week cycle.",
       "basic", (), domain="work"),

    # Adversarial: corrections that must NOT be resolved (10)
    _c("I was going to say Tuesday, but Wednesday works better.",
       "I was going to say Tuesday, but Wednesday works better.", "basic", (), pres=True,
       preserved=("I was going to say Tuesday, but Wednesday works better",),
       notes="adversarial: both dates meaningful"),
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
    _c("we need between 10 and 15 copies of the report",
       "We need between 10 and 15 copies of the report.", "basic", (), pres=True,
       preserved=("between 10 and 15",)),
    _c("the temperature stays between 20 and 25 degrees",
       "The temperature stays between 20 and 25 degrees.", "basic", (), pres=True,
       preserved=("between 20 and 25",)),
    _c("the project runs from January to March",
       "The project runs from January to March.", "basic", (), pres=True,
       preserved=("from January to March",)),
    _c("we need both the budget report and the timeline",
       "We need both the budget report and the timeline.", "basic", (), pres=True,
       preserved=("both the budget report and the timeline",)),
]


# --------------------------------------------------------------------------
# Finalizer (same as V2 stress test)
# --------------------------------------------------------------------------

def _finalize(entry, level, idx):
    """Convert compact entry to full record."""
    record = {
        "id": f"stress_v4_{idx:04d}",
        "input": entry["in"],
        "output": entry["out"],
        "category": entry["cat"],
        "transformations": list(entry["tr"]),
        "num_transformations": len(entry["tr"]),
        "difficulty": "easy" if level == "easy" else ("medium" if level == "medium" else "hard"),
        "domain": entry["domain"],
        "entities": entities_from_text(entry["out"]),
        "corrections": [],
        "reformulations": list(entry["reform"]),
        "formatting": entry["fmt"],
        "preservation_case": entry["pres"],
        "preserved": list(entry["preserved"]),
        "combo": None,
        "template_id": None,
        "seed": idx,
        "generator_version": "v4.0.0",
        "level": level,
        "notes": entry.get("notes"),
    }

    # Build correction metadata
    for abandoned, kept in entry["corr"]:
        # Determine entity kind from spans
        out_spans = entity_spans(entry["out"])
        kind = "unknown"
        for span in out_spans:
            if span.canonical.lower() in kept.lower():
                kind = span.kind
                break
        record["corrections"].append({
            "abandoned": abandoned,
            "kept": kept,
            "kind": kind,
            "marker": "no wait",
            "span": [0, 0],
        })

    return record


ALL_LEVELS_V4 = {
    "easy": EASY_V4,
    "medium": MEDIUM_V4,
    "hard": HARD_V4,
    "extreme": EXTREME_V4,
}


def build_stress_records_v4() -> list[dict]:
    records = []
    idx = 1
    for level in LEVELS:
        entries = ALL_LEVELS_V4[level]
        if len(entries) != EXPECTED_PER_LEVEL_V4:
            raise ValueError(f"V4 {level} has {len(entries)} entries, expected {EXPECTED_PER_LEVEL_V4}")
        for entry in entries:
            records.append(_finalize(entry, level, idx))
            idx += 1
    if len(records) != 200:
        raise ValueError(f"V4 total {len(records)} records, expected 200")
    return records


def validate_stress_v4(records: list[dict]) -> list[str]:
    """Schema-level validation; returns a list of problems (empty = OK)."""
    problems = []
    counts = {level: 0 for level in LEVELS}
    seen_ids = set()
    for r in records:
        if not r["id"].startswith("stress_v4_"):
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
        for sub in r["preserved"]:
            if sub not in r["output"]:
                problems.append(f"preserved {sub!r} missing from output in {r['id']}")
    for level in LEVELS:
        if counts[level] != EXPECTED_PER_LEVEL_V4:
            problems.append(f"{level} has {counts[level]} records, expected {EXPECTED_PER_LEVEL_V4}")
    return problems


if __name__ == "__main__":
    records = build_stress_records_v4()
    problems = validate_stress_v4(records)
    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(records)} records, {len(problems)} problems")
