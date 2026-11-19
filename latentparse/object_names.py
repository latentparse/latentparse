"""The object names of a question: one text-only call on the question and options, no photo. LatentParse uses them for
the hidden consensus boxes (agent.py blend_boxes) and in the Fixed-Sequence mode (agent.py --all-tools), and the tool
calibration uses them too (calib_blend.py)."""
import json
import re

OBJECTS = """You prepare a spatial question for tools that search the image(s) by text. You do not see the image(s).
List the objects or places the question and its options ask about. Reply with JSON only:
{"objects": ["<short noun phrase>", ...]}
At most MAX_OBJECTS objects, most important first, each a short noun phrase as a caption would name it (e.g. "the red car").
Leave out the camera, the viewer ("you", "I"), directions, distances and option letters. If nothing physical is named, reply {"objects": []}."""


def _parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# Names that are not objects in the photo (they hit 0 of 8 boxes in the flip test); dropped in --latents runs.
NOT_OBJECT = re.compile(r"\b(photos?|pictures?|images?|cameras?|viewers?|shooting|frames?)\b", re.I)


def list_objects(chat, item, max_objects, clean=False):
    """Object names from the question text alone (one short call, no photo). clean: drop names of photos, cameras or
    viewers (NOT_OBJECT)."""
    msgs = [{"role": "system", "content": OBJECTS.replace("MAX_OBJECTS", str(max_objects))},
            {"role": "user", "content": [{"type": "text", "text": f"Question: {item['question']}\nOptions: "
                                         + "; ".join(f"{k}. {v}" for k, v in item["options"])}]}]
    reply, _ = chat(msgs, 120)
    parsed = _parse_json(reply)
    objs = parsed.get("objects") if isinstance(parsed, dict) else None
    objs = [str(o).strip()[:60] for o in objs if str(o).strip()] if isinstance(objs, list) else []
    if clean:
        objs = [o for o in objs if not NOT_OBJECT.search(o)]
    return list(dict.fromkeys(objs))[:max_objects], reply
