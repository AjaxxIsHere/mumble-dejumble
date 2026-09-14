"""Clean-output template bank: 600 templates across 12 domains.

Invariants (pinned by tests/test_templates.py):
- Slot kinds are the KNOWN_SLOT_KINDS below; every filled template must extract
  back as entities of the matching kind.
- TECH pool entries are bare ("CI pipeline"), so every template writes
  "the {tech}" / "the {tech_s}" / "the {tech_p}".
- ITEMS pool entries are bare nouns; templates use "the {item}" or bare "{item}".
  Agreement-sensitive positions use {item_s} (singular pool) or {item_p}
  (plural pool) so "the curtains needs updating" can never occur.
- Context-aware numbers: {duration} for delay/duration phrases, {port} and {id}
  for technical/reference slots — generic {number} never exceeds 999.
- Repeated generated kinds (time/date/number) fill DISTINCT values by default;
  pass repeat_same=True (used by preservation templates) to keep one value.
"""

import random
import re
from dataclasses import dataclass

from data.generators.entities import (
    PLURAL_TECH,
    PoolSampler,
    make_date,
    make_duration,
    make_id,
    make_number,
    make_port,
    make_time,
)

KNOWN_SLOT_KINDS = (
    "name", "name2", "item", "item_s", "item_p", "items", "location",
    "tech", "tech_s", "tech_p", "number", "price", "percent", "version",
    "duration", "port", "id", "time", "date",
)

DOMAINS = (
    "everyday", "work", "programming", "university", "messaging", "shopping",
    "travel", "scheduling", "technical", "notes", "email", "lists",
)


@dataclass(frozen=True)
class Template:
    id: str
    domain: str
    text: str
    slots: tuple[str, ...]
    preserved_patterns: tuple[str, ...] = ()
    repeat_same: bool = False


_GENERATED_MAKERS = {
    "number": lambda rng: make_number(rng),
    "price": lambda rng: make_number(rng, "price"),
    "percent": lambda rng: make_number(rng, "percent"),
    "version": lambda rng: make_number(rng, "version"),
    "duration": make_duration,
    "port": make_port,
    "id": make_id,
    "time": make_time,
    "date": make_date,
}

_SLOT_RE = re.compile(r"\{(\w+)\}")


def sample_kind(kind: str, sampler: PoolSampler, rng: random.Random,
                cache: dict[str, str], used: dict[str, set[str]]) -> str:
    """Resolve one slot kind to a value. Shared by fill_slots and longform
    extension filling so both tiers keep identical grammar rules.

    - "name" resolves to ONE person for the whole record; "name2" samples a
      second, distinct person (the sampler draws without replacement).
    - item_s / item_p / tech_s / tech_p draw from the number-restricted pools.
    - Generated kinds dedupe against their per-kind used-set (bounded retries).
    """
    if kind in cache:
        return cache[kind]
    if kind in ("name", "name2"):
        v = sampler.sample("name")
        if rng.random() < 0.3:
            v += " " + sampler.sample("surname")
        if kind == "name":
            cache["name"] = v
    elif kind in ("item", "item_s", "item_p"):
        v = sampler.sample(kind)
    elif kind == "location":
        v = sampler.sample("location")
    elif kind in ("tech", "tech_s", "tech_p"):
        v = sampler.sample(kind)
    elif kind == "items":
        n = rng.randint(2, 4)
        parts = [sampler.sample("item") for _ in range(n)]
        v = ", ".join(parts[:-1]) + " and " + parts[-1] if n > 2 else f"{parts[0]} and {parts[1]}"
    elif kind in _GENERATED_MAKERS:
        maker = _GENERATED_MAKERS[kind]
        v = maker(rng)
        for _ in range(20):
            if v.lower() not in used[kind]:
                break
            v = maker(rng)
        used[kind].add(v.lower())
    else:
        raise ValueError(f"unknown slot kind {kind!r}")
    return v


def fill_slots(
    template: Template,
    sampler: PoolSampler,
    rng: random.Random,
    repeat_same: bool | None = None,
) -> tuple[str, dict[str, list[str]]]:
    """Fill a template's slots, returning (filled_text, slot_mapping).

    - "name" always resolves to one person (repeat occurrences are the same);
      "name2" is a second, distinct person.
    - Other kinds are distinct within a record by default.
    - repeat_same=True caches every kind — used by preservation templates
      where the same value repeating is the point ("Keep 42 as 42").
    """
    repeat_same = template.repeat_same if repeat_same is None else repeat_same
    cache: dict[str, str] = {}
    used: dict[str, set[str]] = {k: set() for k in _GENERATED_MAKERS}

    def value(kind: str) -> str:
        if repeat_same and kind in cache:
            return cache[kind]
        v = sample_kind(kind, sampler, rng, cache, used)
        if repeat_same and kind not in cache:
            cache[kind] = v
        return v

    # One value per slot OCCURRENCE (repeated kinds get distinct values unless
    # repeat_same), filled sequentially so "from {time} to {time}" gets two
    # different times. A dict-keyed .format() would collapse them.
    values = [value(kind) for kind in template.slots]
    it = iter(values)
    filled = _SLOT_RE.sub(lambda m: next(it), template.text)
    mapping: dict[str, list[str]] = {}
    for kind, v in zip(template.slots, values):
        mapping.setdefault(kind, []).append(v)
    return filled, mapping


def fill_pattern(pattern: str, mapping: dict[str, list[str]]) -> str:
    """Fill a preserved pattern, consuming each kind's values in occurrence order."""
    counters = {kind: 0 for kind in mapping}

    def repl(m: re.Match) -> str:
        kind = m.group(1)
        value = mapping[kind][counters[kind]]
        counters[kind] += 1
        return value

    return _SLOT_RE.sub(repl, pattern)


# (id, text, "space-separated slot kinds")
_RAW: dict[str, list[tuple[str, str, str]]] = {
    'everyday': [
        ('everyday_001', 'I need to call {name} today.', 'name'),
        ('everyday_002', "Let's grab lunch at {location}.", 'location'),
        ('everyday_003', 'We should watch a movie tonight.', ''),
        ('everyday_004', 'I have to clean the house this weekend.', ''),
        ('everyday_005', 'Call {name} about the {date} plans.', 'name date'),
        ('everyday_006', 'We need {items} from the store.', 'items'),
        ('everyday_007', 'Dinner with {name} is at {time}.', 'name time'),
        ('everyday_008', "Let's walk in the park after {time}.", 'time'),
        ('everyday_009', "I'm meeting {name} at {location}.", 'name location'),
        ('everyday_010', 'We should visit {name} on {date}.', 'name date'),
        ('everyday_011', 'The laundry needs doing today.', ''),
        ('everyday_012', "Let's cook {item} for dinner.", 'item'),
        ('everyday_013', "I'll call you at {time}.", 'time'),
        ('everyday_014', 'We should try that new place in {location}.', 'location'),
        ('everyday_015', 'Pick up the kids at {time}.', 'time'),
        ('everyday_016', 'The dog needs a walk before {time}.', 'time'),
        ('everyday_017', "Let's meet at {location} at {time}.", 'location time'),
        ('everyday_018', 'I need to pay the bills by {date}.', 'date'),
        ('everyday_019', 'We should get the {item} fixed.', 'item'),
        ('everyday_020', "Let's have {name} over on {date}.", 'name date'),
        ('everyday_021', "I'll be home by {time}.", 'time'),
        ('everyday_022', 'We should book the {item} for the weekend.', 'item'),
        ('everyday_023', "Let's call {name} after dinner.", 'name'),
        ('everyday_024', 'The plants need watering on {date}.', 'date'),
        ('everyday_025', 'We should visit {location} this weekend.', 'location'),
        ('everyday_026', "I'll text you when I get to {location}.", 'location'),
        ('everyday_027', "Let's plan the {date} picnic.", 'date'),
        ('everyday_028', 'We need to renew the {item}.', 'item'),
        ('everyday_029', "Let's meet {name} for coffee at {time}.", 'name time'),
        ('everyday_030', "I'll take the car to {location} today.", 'location'),
        ('everyday_031', 'We should sort the garage this weekend.', ''),
        ('everyday_032', "Let's invite {name} for {date}.", 'name date'),
        ('everyday_033', "I'll pick up {items} after work.", 'items'),
        ('everyday_034', 'We should replace the {item}.', 'item'),
        ('everyday_035', "Let's go to the beach on {date}.", 'date'),
        ('everyday_036', "I'll call the doctor before {time}.", 'time'),
        ('everyday_037', 'We should paint the room this month.', ''),
        ('everyday_038', "Let's have breakfast at {location}.", 'location'),
        ('everyday_039', 'I need to return the {item} by {date}.', 'item date'),
        ('everyday_040', 'We should check on {name} today.', 'name'),
        ('everyday_041', "Let's watch the game at {time}.", 'time'),
        ('everyday_042', "I'll be at {location} until {time}.", 'location time'),
        ('everyday_043', 'We should try {item} from the new recipe.', 'item'),
        ('everyday_044', "Let's invite {name} to {location}.", 'name location'),
        ('everyday_045', "I'll handle it after {time}.", 'time'),
    ],
    'work': [
        ('work_001', 'Send {name} the {item} by {date}.', 'name item date'),
        ('work_002', 'Please share the {item} with {name} before {time}.', 'item name time'),
        ('work_003', 'We should schedule the meeting for {date} at {time}.', 'date time'),
        ('work_004', 'I need {name} to review the {item} today.', 'name item'),
        ('work_005', 'The team is meeting at {location} on {date}.', 'location date'),
        ('work_006', 'Could you update the {tech} before {time}?', 'tech time'),
        ('work_007', 'The {item_s} is due on {date}.', 'item_s date'),
        ('work_008', 'Ask {name} to prepare the {item} for {date}.', 'name item date'),
        ('work_009', 'We moved the deadline to {date} at {time}.', 'date time'),
        ('work_010', 'Remind {name} about the {tech} tomorrow.', 'name tech'),
        ('work_011', 'The client wants the {item} delivered by {date}.', 'item date'),
        ('work_012', "Let's sync with {name} at {time} in {location}.", 'name time location'),
        ('work_013', 'Draft a summary of the {item} for the team.', 'item'),
        ('work_014', 'Forward the {item} to {name} before noon.', 'item name'),
        ('work_015', 'The budget review happens on {date}.', 'date'),
        ('work_016', 'Please print {number} copies of the {item}.', 'number item'),
        ('work_017', 'We need to talk to {name} about the {tech} issue.', 'name tech'),
        ('work_018', 'The quarterly report goes out on {date}.', 'date'),
        ('work_019', 'Book {location} for the workshop on {date}.', 'location date'),
        ('work_020', 'Send the invoice for {price} to {name}.', 'price name'),
        ('work_021', 'The standup will run at {time}.', 'time'),
        ('work_022', 'We should discuss the {tech} with {name}.', 'tech name'),
        ('work_023', 'The offsite is at {location} on {date}.', 'location date'),
        ('work_024', 'Review the {item} and send notes to {name}.', 'item name'),
        ('work_025', 'The vendor meeting starts at {time} on {date}.', 'time date'),
        ('work_026', 'Please confirm the order for {items}.', 'items'),
        ('work_027', 'The hiring panel meets at {time}.', 'time'),
        ('work_028', 'Update the {tech} before the {date} release.', 'tech date'),
        ('work_029', 'Ask {name} to file the {item} by {time}.', 'name item time'),
        ('work_030', 'We need {number} more people for the {date} shift.', 'number date'),
        ('work_031', 'The roadmap review is scheduled for {date}.', 'date'),
        ('work_032', 'Send the agenda to {name} before the call.', 'name'),
        ('work_033', 'The audit of the {tech} starts on {date}.', 'tech date'),
        ('work_034', 'Please archive the {item} after the {date} review.', 'item date'),
        ('work_035', 'The board meeting is at {time} in {location}.', 'time location'),
        ('work_036', 'Let {name} know the {item} arrived.', 'name item'),
        ('work_037', 'We should move the launch to {date}.', 'date'),
        ('work_038', 'The server maintenance is planned for {date} at {time}.', 'date time'),
        ('work_039', 'Print and sign the {item} before {time}.', 'item time'),
        ('work_040', 'Schedule a catch-up with {name} for {date}.', 'name date'),
        ('work_041', 'The contract expires on {date}.', 'date'),
        ('work_042', 'Send the {item} to the office before {time}.', 'item time'),
        ('work_043', 'We need the {tech} upgraded by {date}.', 'tech date'),
        ('work_044', 'Ask {name} to chair the {time} meeting.', 'name time'),
        ('work_045', 'The performance reviews wrap up on {date}.', 'date'),
    ],
    'programming': [
        ('programming_001', 'Update the {tech} to the latest version.', 'tech'),
        ('programming_002', 'Fix the bug in the {tech} before {date}.', 'tech date'),
        ('programming_003', 'Deploy the {tech} to {location}.', 'tech location'),
        ('programming_004', 'Write a test for the {tech} feature.', 'tech'),
        ('programming_005', 'Refactor the {tech} before the {date} release.', 'tech date'),
        ('programming_006', 'Push the {tech} changes to the branch.', 'tech'),
        ('programming_007', 'Run the {tech} on {location}.', 'tech location'),
        ('programming_008', 'The {tech_s} build fails on port {port}.', 'tech_s port'),
        ('programming_009', 'Add logging to the {tech}.', 'tech'),
        ('programming_010', 'Review the {tech} code with {name}.', 'tech name'),
        ('programming_011', 'Merge the {tech} branch before {time}.', 'tech time'),
        ('programming_012', 'The {tech} migration runs on {date}.', 'tech date'),
        ('programming_013', 'Downgrade the {tech_s} to version {version}.', 'tech_s version'),
        ('programming_014', 'Set up the {tech} on the new server.', 'tech'),
        ('programming_015', 'Document the {tech} API.', 'tech'),
        ('programming_016', 'Optimize the {tech} query.', 'tech'),
        ('programming_017', 'The {tech} incident is still open.', 'tech'),
        ('programming_018', 'Schedule the {tech} deployment for {date}.', 'tech date'),
        ('programming_019', 'Pair with {name} on the {tech} task.', 'name tech'),
        ('programming_020', 'Roll back the {tech} release.', 'tech'),
        ('programming_021', 'The {tech_s} dashboards show errors.', 'tech_s'),
        ('programming_022', 'Configure the {tech} for the {date} launch.', 'tech date'),
        ('programming_023', 'The {tech} cluster is down.', 'tech'),
        ('programming_024', 'Commit the {tech} fix before {time}.', 'tech time'),
        ('programming_025', 'Run benchmarks on the {tech}.', 'tech'),
        ('programming_026', 'The {tech_s} pipeline takes {duration}.', 'tech_s duration'),
        ('programming_027', 'Upgrade the {tech} during the {date} window.', 'tech date'),
        ('programming_028', 'Write the {tech} changelog.', 'tech'),
        ('programming_029', 'The {tech_s} backlog has {number} open items.', 'tech_s number'),
        ('programming_030', 'Test the {tech} against the staging data.', 'tech'),
        ('programming_031', 'The {tech} release candidate goes out on {date}.', 'tech date'),
        ('programming_032', 'Debug the {tech} memory leak.', 'tech'),
        ('programming_033', 'Pin the {tech_s} dependency to {version}.', 'tech_s version'),
        ('programming_034', 'The {tech} docs are outdated.', 'tech'),
        ('programming_035', 'Ship the {tech} hotfix before {time}.', 'tech time'),
        ('programming_036', 'The {tech_s} build is green again.', 'tech_s'),
        ('programming_037', 'Clean up the {tech} logs.', 'tech'),
        ('programming_038', 'The {tech_s} sandbox resets at {time}.', 'tech_s time'),
        ('programming_039', 'Assign the {tech} ticket to {name}.', 'tech name'),
        ('programming_040', 'The {tech} rollback completed.', 'tech'),
        ('programming_041', 'Profile the {tech} startup time.', 'tech'),
        ('programming_042', 'The {tech_s} cert expires on {date}.', 'tech_s date'),
        ('programming_043', 'Stand up the {tech} in the {date} sprint.', 'tech date'),
        ('programming_044', 'The {tech_s} uptime is at {percent}.', 'tech_s percent'),
        ('programming_045', 'Archive the {tech_s} changelog.', 'tech_s'),
    ],
    'university': [
        ('university_001', 'The assignment is due on {date}.', 'date'),
        ('university_002', 'Study for the {date} exam.', 'date'),
        ('university_003', 'Meet {name} at the library at {time}.', 'name time'),
        ('university_004', 'The lecture moves to {location} on {date}.', 'location date'),
        ('university_005', 'Submit the {item} before {time}.', 'item time'),
        ('university_006', "The professor's office hours are at {time}.", 'time'),
        ('university_007', 'Review chapter {number} before the test.', 'number'),
        ('university_008', 'The seminar is in {location} at {time}.', 'location time'),
        ('university_009', 'Register for the course by {date}.', 'date'),
        ('university_010', 'The lab session runs until {time}.', 'time'),
        ('university_011', 'Borrow the {item} from the library.', 'item'),
        ('university_012', 'The group project meets on {date}.', 'date'),
        ('university_013', 'Email {name} the {item}.', 'name item'),
        ('university_014', 'The deadline for the thesis is {date}.', 'date'),
        ('university_015', 'The exam covers chapter {number}.', 'number'),
        ('university_016', 'Study at {location} after class.', 'location'),
        ('university_017', 'The workshop starts at {time} on {date}.', 'time date'),
        ('university_018', 'Ask {name} for the lecture notes.', 'name'),
        ('university_019', 'The presentation is on {date} at {time}.', 'date time'),
        ('university_020', 'Submit the form to {location} by {date}.', 'location date'),
        ('university_021', 'The midterm is on {date}.', 'date'),
        ('university_022', 'Meet the study group at {time}.', 'time'),
        ('university_023', 'The tutorial is in {location}.', 'location'),
        ('university_024', 'The paper needs {number} sources.', 'number'),
        ('university_025', 'The class ends at {time}.', 'time'),
        ('university_026', 'Discuss the {item} with {name}.', 'item name'),
        ('university_027', 'The final exam is on {date}.', 'date'),
        ('university_028', 'Check the schedule for {date}.', 'date'),
        ('university_029', 'The lecture notes are in the {item}.', 'item'),
        ('university_030', 'The course registration closes on {date}.', 'date'),
        ('university_031', 'Meet the advisor at {time} on {date}.', 'time date'),
        ('university_032', 'The lab is in {location}.', 'location'),
        ('university_033', 'The essay is {number} words minimum.', 'number'),
        ('university_034', 'Print the {item} before class.', 'item'),
        ('university_035', 'The presentation slides are due by {date}.', 'date'),
        ('university_036', 'Study the {tech} concepts for the exam.', 'tech'),
        ('university_037', 'The group meets at {location} at {time}.', 'location time'),
        ('university_038', 'The reading list has {number} books.', 'number'),
        ('university_039', 'Turn in the {item} by {time}.', 'item time'),
        ('university_040', 'The seminar speaker is {name}.', 'name'),
        ('university_041', 'The lecture hall is in {location}.', 'location'),
        ('university_042', 'The quiz is on {date} at {time}.', 'date time'),
        ('university_043', 'The course starts on {date}.', 'date'),
        ('university_044', 'Prepare for the {date} defense.', 'date'),
        ('university_045', 'The campus shuttle runs until {time}.', 'time'),
    ],
    'messaging': [
        ('messaging_001', "Text {name} that I'm running {duration} late.", 'name duration'),
        ('messaging_002', "Tell {name} I'll call after {time}.", 'name time'),
        ('messaging_003', 'Send a quick message to {name} about the {item}.', 'name item'),
        ('messaging_004', "Let {name} know we're meeting at {location} at {time}.", 'name location time'),
        ('messaging_005', 'Message {name} to bring the {item} tomorrow.', 'name item'),
        ('messaging_006', 'Tell {name} happy birthday from all of us.', 'name'),
        ('messaging_007', 'Text {name} the address of {location}.', 'name location'),
        ('messaging_008', 'Send {name} a reminder about {date}.', 'name date'),
        ('messaging_009', 'Let the group know the plan changed to {date}.', 'date'),
        ('messaging_010', 'Tell {name} the {item_s} is ready for pickup.', 'name item_s'),
        ('messaging_011', 'Message {name} that the {tech_s} is back online.', 'name tech_s'),
        ('messaging_012', 'Send a voice note to {name} about the {date} trip.', 'name date'),
        ('messaging_013', "Text {name} that we're waiting at {location}.", 'name location'),
        ('messaging_014', 'Tell {name} to check the {item} I sent.', 'name item'),
        ('messaging_015', 'Message the team about the {time} standup.', 'time'),
        ('messaging_016', 'Let {name} know I left the {item} at {location}.', 'name item location'),
        ('messaging_017', 'Text {name} the confirmation code {id}.', 'name id'),
        ('messaging_018', "Tell {name} I'll be free after {time}.", 'name time'),
        ('messaging_019', 'Send {name} the photos from {date}.', 'name date'),
        ('messaging_020', 'Message {name} to reschedule to {date}.', 'name date'),
        ('messaging_021', 'Tell {name} the meeting moved to {location}.', 'name location'),
        ('messaging_022', 'Text {name} that dinner is at {time}.', 'name time'),
        ('messaging_023', 'Let {name} know the {item_s} costs {price}.', 'name item_s price'),
        ('messaging_024', 'Send a quick note to {name} about the {tech}.', 'name tech'),
        ('messaging_025', "Tell {name} I can't make it on {date}.", 'name date'),
        ('messaging_026', 'Message {name} the list of {items}.', 'name items'),
        ('messaging_027', 'Text {name} the gate number for {location}.', 'name location'),
        ('messaging_028', "Let {name} know we're on our way from {location}.", 'name location'),
        ('messaging_029', 'Tell {name} to call the {tech} support line.', 'name tech'),
        ('messaging_030', 'Send {name} a link to the {item}.', 'name item'),
        ('messaging_031', 'Message {name} that the package arrives on {date}.', 'name date'),
        ('messaging_032', "Text {name} that I'll join at {time}.", 'name time'),
        ('messaging_033', 'Tell {name} the reservation is under my name.', 'name'),
        ('messaging_034', 'Let {name} know the tickets are for {date}.', 'name date'),
        ('messaging_035', 'Send {name} a reminder to renew the {item}.', 'name item'),
        ('messaging_036', 'Message the group that the venue is {location}.', 'location'),
        ('messaging_037', 'Tell {name} the estimate came to {price}.', 'name price'),
        ('messaging_038', "Text {name} that I'm stuck at {location}.", 'name location'),
        ('messaging_039', 'Let {name} know we pushed it to {date}.', 'name date'),
        ('messaging_040', 'Send {name} the agenda before {time}.', 'name time'),
        ('messaging_041', 'Message {name} to pick up {items} on the way.', 'name items'),
        ('messaging_042', 'Tell {name} the {tech_s} meeting is at {time}.', 'name tech_s time'),
        ('messaging_043', 'Text {name} that the delivery window is {time}.', 'name time'),
        ('messaging_044', "Let {name} know I'll be at {location} by {time}.", 'name location time'),
        ('messaging_045', 'Send {name} a thank-you note for the {item}.', 'name item'),
        ('messaging_046', 'Tell {name} our flight into {location} is delayed until {time}.', 'name location time'),
        ('messaging_047', 'Message {name} that the car is parked at {location}.', 'name location'),
        ('messaging_048', 'Text {name} the gate changed to {location} at {time}.', 'name location time'),
        ('messaging_049', "Let {name} know I'll land at {time} and head straight to {location}.", 'name time location'),
        ('messaging_050', 'Tell {name} to hold the {time} table for {number} people.', 'name time number'),
        ('messaging_051', 'Message {name} that checkout is at {time} on {date}.', 'name time date'),
        ('messaging_052', 'Text {name} the meeting at {location} starts at {time}.', 'name location time'),
        ('messaging_053', 'Let {name} know the {item_s} shipped and lands on {date}.', 'name item_s date'),
        ('messaging_054', 'Tell {name} the delivery driver is {duration} away.', 'name duration'),
        ('messaging_055', "Message {name} I'll be offline from {time} to {time}.", 'name time time'),
        ('messaging_056', 'Text {name} that the reservation moved to {date} at {time}.', 'name date time'),
        ('messaging_057', 'Tell {name} to meet us at {location} by {time}.', 'name location time'),
        ('messaging_058', 'Message {name} the confirmation id is {id}.', 'name id'),
        ('messaging_059', "Let {name} know I'm running {duration} behind.", 'name duration'),
        ('messaging_060', 'Let {name} know the {item_p} are ready.', 'name item_p'),
    ],
    'shopping': [
        ('shopping_001', 'I need to buy {items}.', 'items'),
        ('shopping_002', 'Add {item} to the grocery list.', 'item'),
        ('shopping_003', 'Pick up {items} from {location}.', 'items location'),
        ('shopping_004', "We're out of {item}, please get more.", 'item'),
        ('shopping_005', 'Buy {item} and {item} on the way home.', 'item item'),
        ('shopping_006', 'The {item_s} is on sale at {location}.', 'item_s location'),
        ('shopping_007', 'Grab {items} before {time}.', 'items time'),
        ('shopping_008', 'Get {number} of the {item}.', 'number item'),
        ('shopping_009', "Don't forget the {item} at {location}.", 'item location'),
        ('shopping_010', 'We should stock up on {items}.', 'items'),
        ('shopping_011', 'Order {item} online instead.', 'item'),
        ('shopping_012', 'The {item_s} costs {price} at {location}.', 'item_s price location'),
        ('shopping_013', 'Pick up the {item} after the {date} sale.', 'item date'),
        ('shopping_014', 'Buy {items} for the trip.', 'items'),
        ('shopping_015', 'Add {item} and {item} to the cart.', 'item item'),
        ('shopping_016', 'Get a gift for {name} at {location}.', 'name location'),
        ('shopping_017', 'We need {item} before {date}.', 'item date'),
        ('shopping_018', 'Compare prices on {item} at {location}.', 'item location'),
        ('shopping_019', 'Stock up on {items} this week.', 'items'),
        ('shopping_020', 'Buy the {item} in {location}.', 'item location'),
        ('shopping_021', 'Get {number} kilos of {item}.', 'number item'),
        ('shopping_022', 'The {item_s} is cheaper at {location}.', 'item_s location'),
        ('shopping_023', 'Pick up {items} from {location} on {date}.', 'items location date'),
        ('shopping_024', "We're running low on {item} and {item}.", 'item item'),
        ('shopping_025', "Buy {item} for {name}'s birthday.", 'item name'),
        ('shopping_026', 'Add {items} to the {date} order.', 'items date'),
        ('shopping_027', 'Get a new {item} this weekend.', 'item'),
        ('shopping_028', 'The {item_s} at {location} is out of stock.', 'item_s location'),
        ('shopping_029', 'Order {item} before {time} for next-day delivery.', 'item time'),
        ('shopping_030', 'Buy {items} in bulk.', 'items'),
        ('shopping_031', 'Check the {item} price online.', 'item'),
        ('shopping_032', 'Pick up the {item} I reserved.', 'item'),
        ('shopping_033', 'We need {item} for the recipe.', 'item'),
        ('shopping_034', 'Get {items} for the {date} party.', 'items date'),
        ('shopping_035', 'Buy {item} from {location}.', 'item location'),
        ('shopping_036', 'The {item_s} is on my list.', 'item_s'),
        ('shopping_037', 'Grab {items} at {location} before it closes.', 'items location'),
        ('shopping_038', 'Order {number} of the {item}.', 'number item'),
        ('shopping_039', 'Get a discount on {item} with the app.', 'item'),
        ('shopping_040', "Buy {items} and drop them at {name}'s place.", 'items name'),
        ('shopping_041', 'The {item_s} aisle is in the back.', 'item_s'),
        ('shopping_042', 'Pick up {item} on {date}.', 'item date'),
        ('shopping_043', 'We need {items} for the week.', 'items'),
        ('shopping_044', 'Get {item} in bulk from {location}.', 'item location'),
        ('shopping_045', 'Add {items} to the shared list.', 'items'),
        ('shopping_046', 'The {item_p} are on backorder until {date}.', 'item_p date'),
    ],
    'travel': [
        ('travel_001', 'Book a flight to {location} for {date}.', 'location date'),
        ('travel_002', 'The train to {location} leaves at {time}.', 'location time'),
        ('travel_003', 'Book a hotel in {location} for {number} nights.', 'location number'),
        ('travel_004', 'We fly out of {location} on {date}.', 'location date'),
        ('travel_005', 'The flight lands at {time}.', 'time'),
        ('travel_006', 'Pack {items} for the trip to {location}.', 'items location'),
        ('travel_007', 'The layover is in {location}.', 'location'),
        ('travel_008', 'Book a car from {location} on {date}.', 'location date'),
        ('travel_009', 'We should visit {location} before {date}.', 'location date'),
        ('travel_010', 'The bus to {location} departs at {time}.', 'location time'),
        ('travel_011', 'Get travel insurance before {date}.', 'date'),
        ('travel_012', 'The visa appointment is at {time} on {date}.', 'time date'),
        ('travel_013', 'Pack the {item} in the carry-on.', 'item'),
        ('travel_014', 'We arrive in {location} on {date}.', 'location date'),
        ('travel_015', 'Book a table in {location} for {time}.', 'location time'),
        ('travel_016', 'The ferry to {location} runs until {time}.', 'location time'),
        ('travel_017', 'Reserve seats for the {date} train.', 'date'),
        ('travel_018', 'We need {item} for the flight.', 'item'),
        ('travel_019', 'The return flight is on {date}.', 'date'),
        ('travel_020', 'Check into {location} by {time}.', 'location time'),
        ('travel_021', 'Rent a car in {location} for {number} days.', 'location number'),
        ('travel_022', 'The shuttle leaves at {time} from {location}.', 'time location'),
        ('travel_023', 'Print the boarding pass for the {time} flight.', 'time'),
        ('travel_024', "We're staying near {location}.", 'location'),
        ('travel_025', 'Book an excursion to {location} on {date}.', 'location date'),
        ('travel_026', 'The taxi from {location} costs {price}.', 'location price'),
        ('travel_027', 'Pack {items} for the beach day.', 'items'),
        ('travel_028', 'The check-in closes at {time}.', 'time'),
        ('travel_029', 'Book the {date} train to {location}.', 'date location'),
        ('travel_030', 'We need to be at {location} by {time}.', 'location time'),
        ('travel_031', 'Get a map of {location}.', 'location'),
        ('travel_032', 'The flight from {location} is delayed.', 'location'),
        ('travel_033', 'Book a day trip from {location}.', 'location'),
        ('travel_034', 'We land in {location} at {time}.', 'location time'),
        ('travel_035', 'Bring the {item} for the {date} hike.', 'item date'),
        ('travel_036', 'The hotel is near {location}.', 'location'),
        ('travel_037', 'Reserve a room in {location} for {date}.', 'location date'),
        ('travel_038', 'The tour starts at {time} at {location}.', 'time location'),
        ('travel_039', 'Pack {number} bags for {location}.', 'number location'),
        ('travel_040', 'We should leave for {location} by {time}.', 'location time'),
        ('travel_041', 'Book tickets for the {date} show in {location}.', 'date location'),
        ('travel_042', 'The cruise departs on {date}.', 'date'),
        ('travel_043', 'Meet the guide at {location} at {time}.', 'location time'),
        ('travel_044', 'Get currency before flying to {location}.', 'location'),
        ('travel_045', 'The connecting flight to {location} is at {time}.', 'location time'),
    ],
    'scheduling': [
        ('scheduling_001', 'The meeting is at {time} on {date}.', 'time date'),
        ('scheduling_002', 'Move the appointment to {date}.', 'date'),
        ('scheduling_003', 'We should meet at {location} at {time}.', 'location time'),
        ('scheduling_004', 'The call is scheduled for {time}.', 'time'),
        ('scheduling_005', 'Reschedule the session for {date}.', 'date'),
        ('scheduling_006', 'The event starts at {time} at {location}.', 'time location'),
        ('scheduling_007', 'Block {number} hours for the {date} workshop.', 'number date'),
        ('scheduling_008', 'The meeting runs from {time} to {time}.', 'time time'),
        ('scheduling_009', 'The demo is on {date} at {time}.', 'date time'),
        ('scheduling_010', 'Shift the appointment to {time}.', 'time'),
        ('scheduling_011', 'The interview is at {time} on {date}.', 'time date'),
        ('scheduling_012', 'Book the room for {date} at {time}.', 'date time'),
        ('scheduling_013', 'The training runs until {time}.', 'time'),
        ('scheduling_014', 'We meet on {date} at {time}.', 'date time'),
        ('scheduling_015', 'The sync with {name} is at {time}.', 'name time'),
        ('scheduling_016', 'Move the deadline to {date}.', 'date'),
        ('scheduling_017', 'The presentation is at {time}.', 'time'),
        ('scheduling_018', 'Schedule the review for {date}.', 'date'),
        ('scheduling_019', 'The appointment is at {location} at {time}.', 'location time'),
        ('scheduling_020', 'We should meet before {time}.', 'time'),
        ('scheduling_021', 'The planning session is on {date}.', 'date'),
        ('scheduling_022', 'Shift the call to {time} on {date}.', 'time date'),
        ('scheduling_023', 'The one-on-one with {name} is at {time}.', 'name time'),
        ('scheduling_024', 'Book {location} from {time} to {time}.', 'location time time'),
        ('scheduling_025', 'The meeting agenda is for {date}.', 'date'),
        ('scheduling_026', 'We should reschedule for {date}.', 'date'),
        ('scheduling_027', 'The event ends at {time}.', 'time'),
        ('scheduling_028', 'Plan the kickoff for {date} at {time}.', 'date time'),
        ('scheduling_029', 'The session with {name} is on {date}.', 'name date'),
        ('scheduling_030', 'Move the standup to {time}.', 'time'),
        ('scheduling_031', 'The workshop is at {location} on {date}.', 'location date'),
        ('scheduling_032', 'Schedule a follow-up for {date}.', 'date'),
        ('scheduling_033', 'The meeting is set for {time}.', 'time'),
        ('scheduling_034', 'We should find {duration} on {date}.', 'duration date'),
        ('scheduling_035', 'The gathering is at {location} at {time}.', 'location time'),
        ('scheduling_036', 'Push the review to {date}.', 'date'),
        ('scheduling_037', 'The rehearsal is at {time} on {date}.', 'time date'),
        ('scheduling_038', 'Arrange a call for {time}.', 'time'),
        ('scheduling_039', 'The town hall is on {date}.', 'date'),
        ('scheduling_040', 'Set the timer for {duration}.', 'duration'),
        ('scheduling_041', 'The ceremony starts at {time}.', 'time'),
        ('scheduling_042', 'We should confirm the {date} slot.', 'date'),
        ('scheduling_043', 'The team dinner is at {time} on {date}.', 'time date'),
        ('scheduling_044', 'Schedule the demo with {name} for {date}.', 'name date'),
        ('scheduling_045', 'The venue booking is until {time}.', 'time'),
        ('scheduling_046', "If {date} doesn't work, shift the sync with {name} to {date} at {time}.", 'date name date time'),
        ('scheduling_047', "The {time} slot is taken, so let's use the {time} slot instead.", 'time time'),
        ('scheduling_048', "Cancel the {date} review if {name} can't make it.", 'date name'),
        ('scheduling_049', 'Move the {date} demo to {location} at {time}.', 'date location time'),
        ('scheduling_050', 'We have a conflict at {time}; reschedule with {name} for {date}.', 'time name date'),
        ('scheduling_051', 'If {location} is booked, fall back to the {date} slot.', 'location date'),
        ('scheduling_052', 'Push the kickoff from {time} to {time} on {date}.', 'time time date'),
        ('scheduling_053', 'The {date} slot works for {name}, but not before {time}.', 'date name time'),
        ('scheduling_054', 'Book {location} for {date} and send the invite to {name}.', 'location date name'),
        ('scheduling_055', 'If the {time} call runs long, move the demo to {time}.', 'time time'),
        ('scheduling_056', 'Put {name} down for {date} at {time} in {location}.', 'name date time location'),
        ('scheduling_057', 'Freeze the calendar between {time} and {time} on {date}.', 'time time date'),
        ('scheduling_058', 'The walkthrough with {name} lands on {date} at {time}.', 'name date time'),
        ('scheduling_059', 'Shift the one-on-one with {name} to {date} at {time}.', 'name date time'),
    ],
    'technical': [
        ('technical_001', 'The {tech_s} latency is {number} milliseconds.', 'tech_s number'),
        ('technical_002', 'Reboot the server in {location} at {time}.', 'location time'),
        ('technical_003', 'The {tech_s} certificate expires on {date}.', 'tech_s date'),
        ('technical_004', 'Monitor the {tech} during the {date} rollout.', 'tech date'),
        ('technical_005', 'The {tech} logs show {number} errors.', 'tech number'),
        ('technical_006', 'Scale the {tech} before {time}.', 'tech time'),
        ('technical_007', 'The {tech_s} pods restarted at {time}.', 'tech_s time'),
        ('technical_008', 'Back up the {tech} before the migration.', 'tech'),
        ('technical_009', 'The {tech_s} throughput is {number} requests.', 'tech_s number'),
        ('technical_010', 'Patch the {tech} during the {date} window.', 'tech date'),
        ('technical_011', 'The {tech} connection to {location} dropped.', 'tech location'),
        ('technical_012', 'Increase the {tech} timeout to {number} seconds.', 'tech number'),
        ('technical_013', 'The {tech_s} node in {location} is offline.', 'tech_s location'),
        ('technical_014', 'Fail over to {location} before {time}.', 'location time'),
        ('technical_015', 'The {tech_s} version mismatch is on {number}.', 'tech_s number'),
        ('technical_016', 'Restore the {tech} snapshot from {date}.', 'tech date'),
        ('technical_017', 'The {tech} queue depth hit {number}.', 'tech number'),
        ('technical_018', 'Rebalance the {tech} after the {date} deploy.', 'tech date'),
        ('technical_019', 'The {tech_s} health check fails on port {port}.', 'tech_s port'),
        ('technical_020', 'Freeze the {tech} changes until {date}.', 'tech date'),
        ('technical_021', 'The {tech} cold start takes {number} seconds.', 'tech number'),
        ('technical_022', 'Rotate the {tech_s} keys on {date}.', 'tech_s date'),
        ('technical_023', 'The {tech_s} peak load is at {time}.', 'tech_s time'),
        ('technical_024', 'Sync the {tech} with {location} before {time}.', 'tech location time'),
        ('technical_025', 'The {tech} alarm fired {number} times.', 'tech number'),
        ('technical_026', 'Drain the {tech} traffic to {location}.', 'tech location'),
        ('technical_027', 'The {tech_s} capacity is at {percent}.', 'tech_s percent'),
        ('technical_028', 'Test the {tech} failover on {date}.', 'tech date'),
        ('technical_029', 'The {tech_s} probe from {location} times out.', 'tech_s location'),
        ('technical_030', 'Rebuild the {tech} image before {date}.', 'tech date'),
        ('technical_031', 'The {tech_s} disk usage reached {percent}.', 'tech_s percent'),
        ('technical_032', 'Isolate the {tech} issue to {location}.', 'tech location'),
        ('technical_033', 'The {tech_s} DNS resolves after {number} seconds.', 'tech_s number'),
        ('technical_034', 'Warm the {tech} cache before {time}.', 'tech time'),
        ('technical_035', 'The {tech_s} baseline is {number}.', 'tech_s number'),
        ('technical_036', 'Audit the {tech} access list on {date}.', 'tech date'),
        ('technical_037', 'The {tech_s} build number is {id}.', 'tech_s id'),
        ('technical_038', 'Route the {tech} traffic through {location}.', 'tech location'),
        ('technical_039', 'The {tech_s} idle timeout is {duration}.', 'tech_s duration'),
        ('technical_040', 'Validate the {tech} config before {time}.', 'tech time'),
        ('technical_041', 'The {tech_s} replica count is {number}.', 'tech_s number'),
        ('technical_042', 'Purge the {tech} cache at {time}.', 'tech time'),
        ('technical_043', 'The {tech_s} benchmark runs on {date}.', 'tech_s date'),
        ('technical_044', 'Track the {tech} error rate after {time}.', 'tech time'),
        ('technical_045', 'The {tech_s} failover test is scheduled for {date}.', 'tech_s date'),
        ('technical_046', 'Roll back the {tech_s} deployment in {location} if the error rate exceeds {percent}.', 'tech_s location percent'),
        ('technical_047', 'Escalate the {tech_s} alert to {name} if it fires again after {time}.', 'tech_s name time'),
        ('technical_048', 'Triage the {tech_s} bug against the {date} changelog.', 'tech_s date'),
        ('technical_049', 'Fail over the {tech_s} traffic to {location} before {time}.', 'tech_s location time'),
        ('technical_050', 'Page {name} if the {tech_s} error rate stays above {percent}.', 'name tech_s percent'),
        ('technical_051', 'Snapshot the {tech_s} state before the {date} deploy.', 'tech_s date'),
        ('technical_052', 'Replay the {tech_s} logs from {time} onward.', 'tech_s time'),
        ('technical_053', 'Quarantine the {tech_s} node in {location} until {date}.', 'tech_s location date'),
        ('technical_054', 'Verify the {tech_s} rollback on {location} by {time}.', 'tech_s location time'),
        ('technical_055', 'Mute the {tech_s} alerts during the {date} maintenance window.', 'tech_s date'),
        ('technical_056', 'Open a ticket with {name} about the {tech_s} outage.', 'name tech_s'),
        ('technical_057', 'Soak-test the {tech_s} fix on port {port} for {duration}.', 'tech_s port duration'),
        ('technical_058', 'Restore the {tech_s} backup into {location} before {time}.', 'tech_s location time'),
        ('technical_059', 'Sweep the {tech_s} credentials by {date}.', 'tech_s date'),
        ('technical_060', 'The {tech_p} are out of sync after the {date} deploy.', 'tech_p date'),
    ],
    'notes': [
        ('notes_001', 'Note that the {item_s} needs updating.', 'item_s'),
        ('notes_002', 'Remind myself to buy {items}.', 'items'),
        ('notes_003', 'The meeting notes say {name} will handle it.', 'name'),
        ('notes_004', 'Jot down the number {number}.', 'number'),
        ('notes_005', 'The {item} goes in the {date} folder.', 'item date'),
        ('notes_006', 'Write down the address of {location}.', 'location'),
        ('notes_007', 'Remember the password for the {tech}.', 'tech'),
        ('notes_008', 'The list includes {items}.', 'items'),
        ('notes_009', 'Note the {item} delivery is on {date}.', 'item date'),
        ('notes_010', 'Keep a copy of the {item}.', 'item'),
        ('notes_011', 'The note says the {tech_s} is down.', 'tech_s'),
        ('notes_012', 'Add a reminder for {date} at {time}.', 'date time'),
        ('notes_013', 'The idea about the {tech} is worth exploring.', 'tech'),
        ('notes_014', 'Save the receipt for the {item}.', 'item'),
        ('notes_015', 'The {item} quantity is {number}.', 'item number'),
        ('notes_016', 'Note that {name} prefers {time}.', 'name time'),
        ('notes_017', 'The backup of the {item} is on {date}.', 'item date'),
        ('notes_018', 'Write a memo about the {tech} outage.', 'tech'),
        ('notes_019', 'The checklist has {number} items.', 'number'),
        ('notes_020', 'Remember to renew the {item} by {date}.', 'item date'),
        ('notes_021', 'The sketch of {location} is in the {item}.', 'location item'),
        ('notes_022', 'Note the meeting outcome for {date}.', 'date'),
        ('notes_023', 'The reference number is {id}.', 'id'),
        ('notes_024', 'Add {items} to the shopping note.', 'items'),
        ('notes_025', 'The idea for {date} is to visit {location}.', 'date location'),
        ('notes_026', 'Keep the {item} somewhere safe.', 'item'),
        ('notes_027', 'The reminder for {time} is set.', 'time'),
        ('notes_028', 'Note that the {tech} upgrade is on {date}.', 'tech date'),
        ('notes_029', 'The measurement is {number}.', 'number'),
        ('notes_030', "Write down {name}'s number.", 'name'),
        ('notes_031', 'The {item_s} instructions are in the drawer.', 'item_s'),
        ('notes_032', 'A quick note: the {tech_s} is fixed.', 'tech_s'),
        ('notes_033', 'The plan for {date} is finalized.', 'date'),
        ('notes_034', 'Remember to plant the seeds by {date}.', 'date'),
        ('notes_035', 'The note about the {item} is on the fridge.', 'item'),
        ('notes_036', 'Save the draft about the {tech}.', 'tech'),
        ('notes_037', 'The reminder says {name} owes me {price}.', 'name price'),
        ('notes_038', 'Note the gate code is {id}.', 'id'),
        ('notes_039', 'The {item_s} return window ends on {date}.', 'item_s date'),
        ('notes_040', 'Write down the recipe for {item}.', 'item'),
        ('notes_041', 'The observation about the {tech} is useful.', 'tech'),
        ('notes_042', 'Add a calendar entry for {date}.', 'date'),
        ('notes_043', 'The snippet about the {tech} is saved.', 'tech'),
        ('notes_044', 'Note that {location} closes at {time}.', 'location time'),
        ('notes_045', 'The memo mentions the {item} twice.', 'item'),
        ('notes_046', 'Note that the {item_p} need updating.', 'item_p'),
        ('notes_047', 'The {item_p} are in the second drawer.', 'item_p'),
    ],
    'email': [
        ('email_001', 'Email {name} the {item} by {time}.', 'name item time'),
        ('email_002', 'Draft an email about the {date} meeting.', 'date'),
        ('email_003', 'Send the {item} to {name} today.', 'item name'),
        ('email_004', 'Reply to {name} about the {tech}.', 'name tech'),
        ('email_005', 'The email from {name} mentions {date}.', 'name date'),
        ('email_006', 'Forward the {item} to the team.', 'item'),
        ('email_007', 'Write an email asking for the {item}.', 'item'),
        ('email_008', 'Send the agenda to everyone by {time}.', 'time'),
        ('email_009', 'The newsletter goes out on {date}.', 'date'),
        ('email_010', 'Email the invoice for {price}.', 'price'),
        ('email_011', 'Reply all about the {tech} outage.', 'tech'),
        ('email_012', 'Send the invitation for {date}.', 'date'),
        ('email_013', 'Draft a follow-up to {name}.', 'name'),
        ('email_014', 'The attachment is the {item_s}.', 'item_s'),
        ('email_015', 'Email the summary after the {time} call.', 'time'),
        ('email_016', 'Send the minutes by {date}.', 'date'),
        ('email_017', 'Compose a note about the {item}.', 'item'),
        ('email_018', 'The inbox has {number} unread messages.', 'number'),
        ('email_019', 'Email {name} the login details.', 'name'),
        ('email_020', 'Send the report before {time}.', 'time'),
        ('email_021', 'Draft a thank-you for {name}.', 'name'),
        ('email_022', 'The out-of-office reply is set until {date}.', 'date'),
        ('email_023', 'Email the {tech} team the error logs.', 'tech'),
        ('email_024', 'Send the contract to {name} for review.', 'name'),
        ('email_025', 'The meeting invite is for {date} at {time}.', 'date time'),
        ('email_026', 'Email the updated {item}.', 'item'),
        ('email_027', 'Forward the {tech} announcement.', 'tech'),
        ('email_028', 'Send the reminder about {date}.', 'date'),
        ('email_029', 'Compose the weekly update by {time}.', 'time'),
        ('email_030', 'Email {name} the address of {location}.', 'name location'),
        ('email_031', 'The auto-reply mentions {time}.', 'time'),
        ('email_032', 'Send the {item} to {location}.', 'item location'),
        ('email_033', 'Draft a cancellation for the {date} event.', 'date'),
        ('email_034', 'Email the receipt for the {item}.', 'item'),
        ('email_035', 'The thread with {name} is long.', 'name'),
        ('email_036', 'Send the poll results by {date}.', 'date'),
        ('email_037', 'Write to support about the {tech}.', 'tech'),
        ('email_038', 'Email the team about the {time} change.', 'time'),
        ('email_039', 'The signature block lists {number}.', 'number'),
        ('email_040', 'Send the {item} as an attachment.', 'item'),
        ('email_041', 'Draft the announcement for {date}.', 'date'),
        ('email_042', 'Email the venue about {location}.', 'location'),
        ('email_043', 'The spam filter blocked {number} messages.', 'number'),
        ('email_044', 'Send the follow-up after {time}.', 'time'),
        ('email_045', 'Email {name} the meeting minutes.', 'name'),
    ],
    'lists': [
        ('lists_001', 'Write a list of {items}.', 'items'),
        ('lists_002', 'Add {items} to the list.', 'items'),
        ('lists_003', 'The checklist covers {items}.', 'items'),
        ('lists_004', 'List the steps for the {tech} setup.', 'tech'),
        ('lists_005', 'Write down {items}.', 'items'),
        ('lists_006', 'The shopping list has {items}.', 'items'),
        ('lists_007', 'List the pros and cons of the {item}.', 'item'),
        ('lists_008', 'Write a checklist for the {date} event.', 'date'),
        ('lists_009', 'Add {items} to the packing list.', 'items'),
        ('lists_010', 'The to-do list includes {items}.', 'items'),
        ('lists_011', 'List everything we need for {location}.', 'location'),
        ('lists_012', 'Write down the steps in order.', ''),
        ('lists_013', 'Add {item} to the top of the list.', 'item'),
        ('lists_014', 'The guest list for {date} is ready.', 'date'),
        ('lists_015', 'List the materials for the {item}.', 'item'),
        ('lists_016', 'Write a list of questions for {name}.', 'name'),
        ('lists_017', 'The agenda includes {items}.', 'items'),
        ('lists_018', 'Add {item} and {item} to the order list.', 'item item'),
        ('lists_019', 'List the requirements for the {tech}.', 'tech'),
        ('lists_020', 'Write down the recipe ingredients.', ''),
        ('lists_021', 'Add {items} to the wish list.', 'items'),
        ('lists_022', 'The errand list has {number} entries.', 'number'),
        ('lists_023', 'List the risks for the {date} launch.', 'date'),
        ('lists_024', 'Write a to-do list for {name}.', 'name'),
        ('lists_025', 'Add {items} to the inventory.', 'items'),
        ('lists_026', 'The packing list includes {items}.', 'items'),
        ('lists_027', 'List the tasks for {date}.', 'date'),
        ('lists_028', 'Write down {number} ideas.', 'number'),
        ('lists_029', 'Add {items} to the grocery order.', 'items'),
        ('lists_030', 'The reading list has {items}.', 'items'),
        ('lists_031', 'List the steps for the {tech} upgrade.', 'tech'),
        ('lists_032', 'Write a bucket list for {location}.', 'location'),
        ('lists_033', 'Add {items} to the weekly plan.', 'items'),
        ('lists_034', 'The criteria list includes the {item}.', 'item'),
        ('lists_035', 'List the snacks for the {date} trip.', 'date'),
        ('lists_036', 'Write down the chores for this week.', ''),
        ('lists_037', 'Add {items} to the project list.', 'items'),
        ('lists_038', 'The playlist has {number} songs.', 'number'),
        ('lists_039', 'List the supplies for the {item}.', 'item'),
        ('lists_040', 'Write a list for {name}.', 'name'),
        ('lists_041', 'Add {items} to the menu.', 'items'),
        ('lists_042', 'The requirement list has {number} items.', 'number'),
        ('lists_043', 'List the stops on the way to {location}.', 'location'),
        ('lists_044', 'Write down {items} in order.', 'items'),
        ('lists_045', 'Add {item} to the do-not-forget list.', 'item'),
        ('lists_046', 'Create a checklist for {name} with {items} and {items}.', 'name items items'),
        ('lists_047', 'Break the plan into {items} first and {items} after lunch.', 'items items'),
        ('lists_048', 'Write the action items as {items}.', 'items'),
        ('lists_049', 'Split the tasks between {name} and {name2}.', 'name name2'),
        ('lists_050', 'Tag {name2} for the {items} part and {name} for the {items} part.', 'name2 items name items'),
        ('lists_051', 'Outline the {date} agenda: {items}.', 'date items'),
        ('lists_052', 'Make two lists: {items} for now and {items} for {date}.', 'items items date'),
        ('lists_053', 'Number the steps for the {tech_s} rollout: {items}.', 'tech_s items'),
        ('lists_054', "Add {items} to {name}'s board and {items} to {name2}'s.", 'items name items name2'),
        ('lists_055', 'Convert the notes into {items}.', 'items'),
        ('lists_056', 'Pair {items} with {items} on the {date} plan.', 'items items date'),
        ('lists_057', 'List the handoff items for {name}: {items}.', 'name items'),
    ],
}

TEMPLATES: dict[str, list[Template]] = {
    domain: [
        Template(id=tid, domain=domain, text=text, slots=tuple(slots.split()))
        for tid, text, slots in rows
    ]
    for domain, rows in _RAW.items()
}

ALL_TEMPLATES = [t for domain in DOMAINS for t in TEMPLATES[domain]]


def templates_for(
    domain: str,
    required_kinds: frozenset[str] = frozenset(),
) -> list[Template]:
    """Templates for a domain that contain every required slot kind.

    Correction plans use this to structurally guarantee a suitable slot exists,
    so "correction with no suitable slot" cannot occur.
    """
    return [t for t in TEMPLATES[domain] if required_kinds <= set(t.slots)]


# Preservation templates: clean outputs whose meaningful repetition/ranges MUST
# survive cleaning (spec §7). preserved_patterns are mini-templates over the same
# slot kinds, rendered with the same mapping, and checked verbatim by V10 and
# the preservation metric.
_PRESERVATION_RAW: list[tuple[str, str, str, tuple[str, ...], bool]] = [
    ('pres_001', 'The meeting runs from {time} to {time}.', 'time time', ('from {time} to {time}',), False),
    ('pres_002', 'We will review the {item} between {date} and {date}.', 'item date date', ('between {date} and {date}',), False),
    ('pres_003', 'It was really, really important to fix the {tech}.', 'tech', ('really, really',), False),
    ('pres_004', 'I need to buy {items}.', 'items', ('{items}',), False),
    ('pres_005', 'Check the numbers twice, once now and once on {date}.', 'date', ('twice, once now and once on {date}',), False),
    ('pres_006', 'We moved the meeting from {time} to {time}.', 'time time', ('from {time} to {time}',), False),
    ('pres_007', 'Send {items} and then send them again on {date}.', 'items date', ('Send {items} and then send them again',), False),
    ('pres_008', 'The count stays {number} until the audit is done.', 'number', ('stays {number}',), False),
    ('pres_009', "The gate code is {number}, don't change it.", 'number', ('The gate code is {number}',), False),
    ('pres_010', 'Call me before {time} or after {time}.', 'time time', ('before {time} or after {time}',), False),
    ('pres_011', 'Write it down, then write it down again on {date}.', 'date', ('Write it down, then write it down again',), False),
    ('pres_012', 'The shuttle goes from {location} to the city center.', 'location', ('from {location} to the city center',), False),
    ('pres_013', 'Read the {item_s} instructions twice before starting.', 'item_s', ('Read the {item_s} instructions twice',), False),
    ('pres_014', 'The password is {number} on both accounts.', 'number', ('is {number} on both accounts',), False),
    ('pres_015', "I'll say it again: the meeting is on {date} at {time}.", 'date time', ("I'll say it again"), False),
    ('pres_016', 'The meeting is on {date}, I repeat, on {date}.', 'date date', ('I repeat, on {date}',), True),
    ('pres_017', 'We need {item}, and more {item}.', 'item item', ('and more {item}',), True),
]

PRESERVATION_TEMPLATES: list[Template] = [
    Template(
        id=tid, domain="notes", text=text, slots=tuple(slots.split()),
        preserved_patterns=patterns, repeat_same=repeat_same,
    )
    for tid, text, slots, patterns, repeat_same in _PRESERVATION_RAW
]
