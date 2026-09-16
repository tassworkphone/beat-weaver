"""Parse Beat Saber v4 format beatmaps into normalized data structures.

V4 maps use an index-based dereferencing pattern: note/bomb/obstacle arrays
contain beat timing and an index into separate data arrays that hold the
position, color, and direction information.
"""

from beat_weaver.schemas.normalized import Arc, Bomb, Note, Obstacle


def parse_v4_notes(beatmap: dict, bpm: float) -> tuple[list[Note], list[Bomb]]:
    """Parse notes and bombs from a v4 beatmap.

    Args:
        beatmap: Parsed JSON dict of a v4 difficulty file.
        bpm: Beats per minute for time conversion.

    Returns:
        Tuple of (notes sorted by beat, bombs sorted by beat).
    """
    notes: list[Note] = []
    bombs: list[Bomb] = []

    color_notes = beatmap.get("colorNotes", [])
    color_notes_data = beatmap.get("colorNotesData", [])

    for raw in color_notes:
        beat = raw.get("b", 0.0)
        idx = raw.get("i", 0)
        if idx < 0 or idx >= len(color_notes_data):
            continue
        data = color_notes_data[idx]
        time_seconds = beat * 60.0 / bpm
        notes.append(Note(
            beat=beat,
            time_seconds=time_seconds,
            x=data.get("x", 0),
            y=data.get("y", 0),
            color=data.get("c", 0),
            cut_direction=data.get("d", 0),
            angle_offset=data.get("a", 0),
        ))

    bomb_notes = beatmap.get("bombNotes", [])
    bomb_notes_data = beatmap.get("bombNotesData", [])

    for raw in bomb_notes:
        beat = raw.get("b", 0.0)
        idx = raw.get("i", 0)
        if idx < 0 or idx >= len(bomb_notes_data):
            continue
        data = bomb_notes_data[idx]
        time_seconds = beat * 60.0 / bpm
        bombs.append(Bomb(
            beat=beat,
            time_seconds=time_seconds,
            x=data.get("x", 0),
            y=data.get("y", 0),
        ))

    notes.sort(key=lambda n: n.beat)
    bombs.sort(key=lambda b: b.beat)
    return notes, bombs


def parse_v4_obstacles(beatmap: dict, bpm: float) -> list[Obstacle]:
    """Parse obstacles from a v4 beatmap.

    Args:
        beatmap: Parsed JSON dict of a v4 difficulty file.
        bpm: Beats per minute for time conversion.

    Returns:
        List of obstacles sorted by beat.
    """
    obstacles: list[Obstacle] = []

    obstacle_entries = beatmap.get("obstacles", [])
    obstacles_data = beatmap.get("obstaclesData", [])

    for raw in obstacle_entries:
        beat = raw.get("b", 0.0)
        idx = raw.get("i", 0)
        if idx < 0 or idx >= len(obstacles_data):
            continue
        data = obstacles_data[idx]
        time_seconds = beat * 60.0 / bpm
        obstacles.append(Obstacle(
            beat=beat,
            time_seconds=time_seconds,
            duration_beats=data.get("d", 0.0),
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("w", 1),
            height=data.get("h", 5),
        ))

    obstacles.sort(key=lambda o: o.beat)
    return obstacles


def parse_v4_arcs(beatmap: dict, bpm: float) -> list[Arc]:
    """Parse arcs from a v4 beatmap.

    V4 arcs reference their head/tail position+color+cutDirection indirectly:
    `hi`/`ti` index into `colorNotesData` (the same array notes dereference into,
    not the `colorNotes` timing array), and an optional `ai` indexes into
    `arcsData` for the multiplier/anchor-mode fields (defaulting to index 0,
    since most arcs share the default shape and only deviations get their own
    `arcsData` entry).

    Args:
        beatmap: Parsed JSON dict of a v4 difficulty file.
        bpm: Beats per minute for time conversion.

    Returns:
        List of arcs sorted by head beat.
    """
    arcs: list[Arc] = []

    arc_entries = beatmap.get("arcs", [])
    arcs_data = beatmap.get("arcsData", [])
    notes_data = beatmap.get("colorNotesData", [])

    for raw in arc_entries:
        head_idx = raw.get("hi", -1)
        if head_idx < 0 or head_idx >= len(notes_data):
            continue
        head_data = notes_data[head_idx]

        tail_idx = raw.get("ti", head_idx)
        tail_data = notes_data[tail_idx] if 0 <= tail_idx < len(notes_data) else head_data

        arc_idx = raw.get("ai", 0)
        extra = arcs_data[arc_idx] if 0 <= arc_idx < len(arcs_data) else {}

        head_beat = raw.get("hb", 0.0)
        tail_beat = raw.get("tb", head_beat)
        arcs.append(Arc(
            beat=head_beat,
            time_seconds=head_beat * 60.0 / bpm,
            x=head_data.get("x", 0),
            y=head_data.get("y", 0),
            color=head_data.get("c", 0),
            cut_direction=head_data.get("d", 0),
            head_multiplier=extra.get("m", 1.0),
            tail_beat=tail_beat,
            tail_time_seconds=tail_beat * 60.0 / bpm,
            tail_x=tail_data.get("x", 0),
            tail_y=tail_data.get("y", 0),
            tail_cut_direction=tail_data.get("d", 0),
            tail_multiplier=extra.get("tm", 1.0),
            mid_anchor_mode=extra.get("a", 0),
        ))

    arcs.sort(key=lambda a: a.beat)
    return arcs
