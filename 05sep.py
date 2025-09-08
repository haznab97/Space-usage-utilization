import streamlit as st
import plotly.graph_objects as go
import copy
import json
import pandas as pd
from collections import Counter, defaultdict
from functools import lru_cache

# ==============================
# ULD definitions
# ==============================
ULD_SPECS = {
    "AKE": {"length": 142, "width": 139, "height": 159, "max_weight": 1500},
    "LD7": {"length": 300, "width": 240, "height": 162, "max_weight": 5000},
    "PLA": {"length": 307, "width": 142, "height": 162, "max_weight": 3174},
}
VOLUME_UNITS = {"AKE": 4.0, "LD7": 10.0, "PLA": 8.0}  # used as cost proxy in greedy planner
ULD_ORDER = ["AKE", "LD7", "PLA"]

# Colors: ULD outlines by type; boxes one standard color
MIXED_COLORS = {"AKE": "#000000", "LD7": "#00aa00", "PLA": "#cc0000"}
BOX_COLOR = "#1f77b4"

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Estimation Load Plan")

# ==============================
# Sidebar (only core settings)
# ==============================
st.sidebar.header("Settings")
run_mixed = st.sidebar.checkbox("Enable Mixed ULD mode", value=True)
# keep within the requested tight range for spacing
default_spacing_cm = st.sidebar.slider("Default ULD spacing (cm) when visualizing", 200, 500, 250, 10)
MAX_BOXES_TO_DRAW = st.sidebar.number_input(
    "Max boxes to draw in 3D",
    min_value=50, max_value=5000, value=400, step=50,
    help="Limit 3D drawing for performance."
)

# ==============================
# Number of cargo types (outside form so it updates immediately)
# ==============================
num_items = st.number_input("Number of cargo types", min_value=1, max_value=20, value=1, key="num_items")

# ==============================
# Input form
# ==============================
with st.form("cargo_input"):
    cargo_data = []
    for i in range(num_items):
        st.markdown(f"**Cargo {i+1}**")
        length = st.number_input(f"Length (cm) [{i+1}]", min_value=1, value=100, key=f"l_{i}")
        width  = st.number_input(f"Width (cm)  [{i+1}]", min_value=1, value=100, key=f"w_{i}")
        height = st.number_input(f"Height (cm) [{i+1}]", min_value=1, value=100, key=f"h_{i}")
        weight = st.number_input(f"Weight (kg) [{i+1}]", min_value=1, value=100, key=f"wt_{i}")
        quantity = st.number_input(f"Quantity [{i+1}]", min_value=1, value=1, key=f"q_{i}")
        cargo_data.append({
            "length": length, "width": width, "height": height,
            "weight": weight, "quantity": quantity
        })
    submitted = st.form_submit_button("Simulate")

# ==============================
# Packer (FFD shelf; upright; L/W swap allowed)
# ==============================
def pack_boxes_ffd_shelf(boxes, uld_spec):
    L, W, H, WMAX = uld_spec["length"], uld_spec["width"], uld_spec["height"], uld_spec["max_weight"]
    boxes = sorted(boxes, key=lambda b: (b["length"] * b["width"], b["height"]), reverse=True)

    packed = []
    uld_count = 0
    remaining = boxes

    while remaining:
        used_weight = 0
        placed_indices = set()
        layer_z = 0
        placed_any = False

        while layer_z < H:
            row_y = 0
            layer_height = 0
            placed_in_layer = False

            while row_y < W:
                x = 0
                row_depth = 0
                row_height = 0
                placed_in_row = False

                for i, box in enumerate(remaining):
                    if i in placed_indices:
                        continue

                    for bl, bw in ((box["length"], box["width"]), (box["width"], box["length"])):
                        if (bl <= L - x and
                            bw <= W - row_y and
                            box["height"] <= H - layer_z and
                            used_weight + box["weight"] <= WMAX):

                            pb = {
                                "length": bl, "width": bw, "height": box["height"],
                                "weight": box["weight"],
                                "x": x, "y": row_y, "z": layer_z,
                                "uld_id": uld_count
                            }
                            packed.append(pb)
                            placed_indices.add(i)

                            x += bl
                            row_depth = max(row_depth, bw)
                            row_height = max(row_height, box["height"])
                            layer_height = max(layer_height, row_height)
                            used_weight += box["weight"]

                            placed_in_row = placed_in_layer = placed_any = True
                            break

                if not placed_in_row:
                    break
                row_y += row_depth

            if not placed_in_layer:
                break
            layer_z += max(1, layer_height)

        if not placed_any:
            return None, float("inf")

        remaining = [b for i, b in enumerate(remaining) if i not in placed_indices]
        uld_count += 1

    return packed, uld_count

# ==============================
# Speed helpers: hashable multiset key + cached packer
# ==============================
def boxes_key_tuple(boxes):
    return tuple(sorted((b["length"], b["width"], b["height"], b["weight"]) for b in boxes))

@lru_cache(maxsize=4096)
def _pack_ffd_cached(uld_name, boxes_key):
    spec = ULD_SPECS[uld_name]
    bxs = [{"length": l, "width": w, "height": h, "weight": wt} for (l, w, h, wt) in boxes_key]
    return pack_boxes_ffd_shelf(bxs, spec)

# ==============================
# Mixed optimizer (fast greedy + caching)
# ==============================
def pack_mixed_optimizer_fast(boxes, uld_specs, unit_costs):
    remaining = list(boxes)
    plan_instances = []
    total_units = 0.0
    if not remaining:
        return [], 0.0, []

    while remaining:
        key = boxes_key_tuple(remaining)
        best = None  # (score, name, packed, cnt, spec)
        for name, spec in uld_specs.items():
            packed, cnt = _pack_ffd_cached(name, key)
            if not packed or cnt == float("inf"):
                continue
            units = cnt * unit_costs[name]
            score = (len(packed) / units) if units > 0 else 0.0
            if best is None or score > best[0]:
                best = (score, name, packed, cnt, spec)

        if best is None:
            break

        _, name, packed, cnt, spec = best

        groups = defaultdict(list)
        for p in packed:
            groups[p["uld_id"]].append({**p})
        for _, grp in sorted(groups.items()):
            for g in grp:
                g["uld_id"] = 0
            plan_instances.append({"type": name, "spec": spec, "packed": grp})

        total_units += cnt * unit_costs[name]

        # remove packed by multiset
        taken = defaultdict(int)
        for p in packed:
            taken[(p["length"], p["width"], p["height"], p["weight"])] += 1
        new_remaining = []
        for b in remaining:
            k = (b["length"], b["width"], b["height"], b["weight"])
            if taken.get(k, 0) > 0:
                taken[k] -= 1
            else:
                new_remaining.append(b)
        if len(new_remaining) == len(remaining):  # safety
            break
        remaining = new_remaining

    placed_all = []
    for idx, inst in enumerate(plan_instances):
        for p in inst["packed"]:
            q = dict(p); q["uld_idx"] = idx
            placed_all.append(q)
    return plan_instances, total_units, placed_all

# ==============================
# Viz helpers
# ==============================
EDGES = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]

def draw_box_wireframe(fig, corners, color, width=2, showlegend=False, name=None):
    for e in EDGES:
        fig.add_trace(go.Scatter3d(
            x=[corners[e[0]][0], corners[e[1]][0]],
            y=[corners[e[0]][1], corners[e[1]][1]],
            z=[corners[e[0]][2], corners[e[1]][2]],
            mode="lines", line=dict(color=color, width=width),
            showlegend=showlegend, name=name
        ))

# ==============================
# NEW: normalize so 120x100x100 == 100x120x100
# ==============================
def normalize_boxes(boxes):
    """Force (length >= width) so base-rotation inputs are equivalent."""
    out = []
    for b in boxes:
        l, w = b["length"], b["width"]
        if w > l:
            l, w = w, l
        out.append({**b, "length": l, "width": w})
    return out

# ==============================
# Run simulation
# ==============================
if submitted:
    # expand all boxes
    all_boxes = []
    for item in cargo_data:
        for _ in range(item["quantity"]):
            all_boxes.append({
                "length": item["length"], "width": item["width"],
                "height": item["height"], "weight": item["weight"]
            })

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Normalize so base rotation is order-invariant (ONLY CHANGE)
    all_boxes = normalize_boxes(all_boxes)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Early reject: items that can't fit any ULD (saves time)
    def fits_any_uld(b):
        for s in ULD_SPECS.values():
            if b["height"] <= s["height"] and b["weight"] <= s["max_weight"]:
                if ((b["length"] <= s["length"] and b["width"] <= s["width"]) or
                    (b["width"] <= s["length"] and b["length"] <= s["width"])):
                    return True
        return False

    filtered = [b for b in all_boxes if fits_any_uld(b)]
    dropped = len(all_boxes) - len(filtered)
    all_boxes = filtered
    if dropped:
        st.warning(f"{dropped} item(s) cannot fit in any ULD and were excluded.")

    # Single-ULD results (cached)
    single_results = {}
    single_counts = {}
    key_all = boxes_key_tuple(all_boxes)
    for name, spec in ULD_SPECS.items():
        packed, cnt = _pack_ffd_cached(name, key_all)
        if packed:
            single_results[name] = (packed, cnt, spec)
            single_counts[name] = cnt
    st.session_state.single_type_results = single_results
    st.session_state.single_counts = single_counts

    # Mixed results (fast)
    if run_mixed:
        instances, total_units, placed_all = pack_mixed_optimizer_fast(all_boxes, ULD_SPECS, VOLUME_UNITS)
        st.session_state.mixed = {"instances": instances, "total_units": total_units, "placed_all": placed_all} if instances else None
    else:
        st.session_state.mixed = None

# ==============================
# Display results (with 3D-on-demand buttons)
# ==============================
if "single_type_results" in st.session_state or "mixed" in st.session_state:
    tab_single, tab_mixed = st.tabs(["Single ULD results", "Mixed ULD plan"])

    # Persist button states in session_state
    if "show3d_single" not in st.session_state: st.session_state.show3d_single = False
    if "show3d_mixed"  not in st.session_state: st.session_state.show3d_mixed  = False

    # --- Single ULD tab ---
    with tab_single:
        counts = st.session_state.get("single_counts", {})
        if counts:
            used_types = [t for t in ULD_ORDER if t in counts]
            df = pd.DataFrame(
                [
                    [t for t in used_types],            # Row 1
                    [counts[t] for t in used_types],    # Row 2
                ],
                index=["Type of ULD", "Amount needed"],
                columns=used_types
            ).astype(str)
            st.subheader("Single-ULD results (FFD)")
            st.table(df)

            choice = st.selectbox("Select ULD to visualize (3D available on demand)", used_types)
            chosen_result, cnt, spec = st.session_state.single_type_results[choice]
            st.info(f"{choice}: {cnt} used")

            # Button to show 3D
            if st.button("Show 3D layout (Single ULD)"):
                st.session_state.show3d_single = True

            if st.session_state.show3d_single:
                gap = st.slider("3D spacing (cm)", 300, 500, default_spacing_cm, 10, key="gap_single")
                OFFSET = gap

                draw_iter = chosen_result
                if len(draw_iter) > MAX_BOXES_TO_DRAW:
                    st.caption(f"Showing first {MAX_BOXES_TO_DRAW}/{len(draw_iter)} boxes for performance.")
                    draw_iter = draw_iter[:MAX_BOXES_TO_DRAW]

                fig = go.Figure()

                # ULD frames
                for uld_id in sorted(set(b["uld_id"] for b in draw_iter)):
                    x0 = uld_id * OFFSET
                    x1 = x0 + spec["length"]
                    y1, z1 = spec["width"], spec["height"]
                    corners = [[x0,0,0],[x1,0,0],[x1,y1,0],[x0,y1,0],
                               [x0,0,z1],[x1,0,z1],[x1,y1,z1],[x0,y1,z1]]
                    draw_box_wireframe(fig, corners, color="#666666", width=3)

                # Boxes
                for b in draw_iter:
                    x = b["x"] + b["uld_id"] * OFFSET
                    y, z = b["y"], b["z"]
                    l, w, h = b["length"], b["width"], b["height"]
                    corners = [[x,y,z],[x+l,y,z],[x+l,y+w,z],[x,y+w,z],
                               [x,y,z+h],[x+l,y,z+h],[x+l,y+w,z+h],[x,y+w,z+h]]
                    draw_box_wireframe(fig, corners, color=BOX_COLOR, width=2)

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title="Length (cm)"),
                        yaxis=dict(title="Width (cm)"),
                        zaxis=dict(title="Height (cm)"),
                        aspectmode="data"
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No single-type packing feasible with current inputs.")

    # --- Mixed ULD tab ---
    with tab_mixed:
        mix = st.session_state.get("mixed")
        if mix:
            type_counts = Counter(inst["type"] for inst in mix["instances"])
            used_types = [t for t in ULD_ORDER if type_counts[t] > 0]
            if used_types:
                df = pd.DataFrame(
                    [
                        [t for t in used_types],
                        [type_counts[t] for t in used_types],
                    ],
                    index=["Type of ULD", "Amount needed"],
                    columns=used_types
                ).astype(str)
                st.subheader("Mixed ULD plan")
                st.table(df)
            else:
                st.info("No ULDs used for the current inputs.")

            # Button to show 3D
            if st.button("Show 3D layout (Mixed ULD plan)"):
                st.session_state.show3d_mixed = True

            if st.session_state.show3d_mixed and mix.get("instances"):
                gap = st.slider("3D spacing (cm)", 300, 500, default_spacing_cm, 10, key="gap_mixed")
                OFFSET = gap

                draw_iter = mix["placed_all"]
                if len(draw_iter) > MAX_BOXES_TO_DRAW:
                    st.caption(f"Showing first {MAX_BOXES_TO_DRAW}/{len(draw_iter)} boxes for performance.")
                    draw_iter = draw_iter[:MAX_BOXES_TO_DRAW]

                fig = go.Figure()

                # ULD outlines by type
                for idx, inst in enumerate(mix["instances"]):
                    spec = inst["spec"]
                    color = MIXED_COLORS.get(inst["type"], "#666666")
                    x0 = idx * OFFSET
                    x1 = x0 + spec["length"]
                    y1, z1 = spec["width"], spec["height"]
                    corners = [[x0,0,0],[x1,0,0],[x1,y1,0],[x0,y1,0],
                               [x0,0,z1],[x1,0,z1],[x1,y1,z1],[x0,y1,z1]]
                    draw_box_wireframe(fig, corners, color=color, width=4, showlegend=True, name=f"{inst['type']} #{idx}")

                # Boxes
                for b in draw_iter:
                    if "uld_idx" not in b or b["uld_idx"] < 0 or b["uld_idx"] >= len(mix["instances"]):
                        continue
                    x = b["x"] + b["uld_idx"] * OFFSET
                    y, z = b["y"], b["z"]
                    l, w, h = b["length"], b["width"], b["height"]
                    corners = [[x,y,z],[x+l,y,z],[x+l,y+w,z],[x,y+w,z],
                               [x,y,z+h],[x+l,y,z+h],[x+l,y+w,z+h],[x,y+w,z+h]]
                    draw_box_wireframe(fig, corners, color=BOX_COLOR, width=2)

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title="Length (cm)"),
                        yaxis=dict(title="Width (cm)"),
                        zaxis=dict(title="Height (cm)"),
                        aspectmode="data"
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=700,
                    legend=dict(itemsizing="constant")
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enable Mixed ULD mode and click **Simulate**.")
