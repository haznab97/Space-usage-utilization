import streamlit as st
import plotly.graph_objects as go
import copy
import json
import pandas as pd
from collections import Counter, defaultdict
from functools import lru_cache

# ==============================
# ULD definitions (latest specs)
# ==============================
ULD_SPECS = {
    "AKE": {"length": 142, "width": 139, "height": 159, "max_weight": 1500},  # 4 cbm
    "LD7": {"length": 300, "width": 240, "height": 162, "max_weight": 5000},  # 10 cbm
    "PLA": {"length": 307, "width": 142, "height": 162, "max_weight": 3174},  # 8 cbm
}
VOLUME_UNITS = {"AKE": 4.0, "LD7": 10.0, "PLA": 8.0}  # internal use only
ULD_ORDER = ["AKE", "LD7", "PLA"]  # display order

# Colors: ULD outlines by type; boxes one standard color everywhere
MIXED_COLORS = {"AKE": "#000000", "LD7": "#00aa00", "PLA": "#cc0000"}  # outlines
BOX_COLOR = "#1f77b4"  # standardized box color

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Estimation Load Plan")

# ==============================
# Sidebar
# ==============================
st.sidebar.header("Settings")
run_mixed = st.sidebar.checkbox("Enable Mixed ULD mode", value=True)
gap_mixed_default = st.sidebar.slider("Mixed mode spacing (cm)", 200, 300, 250, 10)

# ==============================
# Input form
# ==============================
with st.form("cargo_input"):
    num_items = st.number_input("Number of cargo types", min_value=1, max_value=20, value=1)
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
# FFD Shelf Packer (upright only, L/W rotation allowed)
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
# Mixed-Type Optimizer (recursive search)
# ==============================
def pack_mixed_optimizer(boxes, uld_specs, unit_costs):
    def pack_into_type(bxs, type_name):
        spec = uld_specs[type_name]
        packed, cnt = pack_boxes_ffd_shelf(copy.deepcopy(bxs), spec)
        if not packed or cnt == float("inf"):
            return None, float("inf")
        groups = defaultdict(list)
        for p in packed:
            groups[p["uld_id"]].append({**p})
        instances = []
        for _, grp in sorted(groups.items()):
            for g in grp:
                g["uld_id"] = 0
            instances.append({"type": type_name, "spec": spec, "packed": grp})
        units = cnt * unit_costs[type_name]
        return instances, units

    def key_fn(bxs):
        return json.dumps(sorted(bxs, key=lambda x: (x["length"], x["width"], x["height"], x["weight"])))

    @lru_cache(maxsize=None)
    def search(bxs_json):
        bxs = json.loads(bxs_json)
        if not bxs:
            return [], 0.0

        best_plan, best_units = None, float("inf")

        for name in uld_specs.keys():
            insts, units = pack_into_type(bxs, name)
            if insts is not None and units < best_units:
                best_plan, best_units = insts, units

        n = len(bxs)
        if n > 1:
            for i in range(1, n):
                left, right = bxs[:i], bxs[i:]
                lp, lu = search(key_fn(left))
                rp, ru = search(key_fn(right))
                if lp is not None and rp is not None:
                    total = lu + ru
                    if total < best_units:
                        best_plan, best_units = lp + rp, total

        return best_plan, best_units

    plan, total_units = search(key_fn(boxes))
    if plan is None:
        return None, float("inf"), None

    placed_all = []
    for idx, inst in enumerate(plan):
        for p in inst["packed"]:
            q = dict(p)
            q["uld_idx"] = idx
            placed_all.append(q)
    return plan, total_units, placed_all

# ==============================
# Run simulation
# ==============================
if submitted:
    all_boxes = []
    for item in cargo_data:
        for _ in range(item["quantity"]):
            all_boxes.append({
                "length": item["length"], "width": item["width"],
                "height": item["height"], "weight": item["weight"]
            })

    # Single-ULD results (counts per type)
    single_results = {}
    single_counts = {}  # name -> count
    for name, spec in ULD_SPECS.items():
        packed, cnt = pack_boxes_ffd_shelf(copy.deepcopy(all_boxes), spec)
        if packed:
            single_results[name] = (packed, cnt, spec)
            single_counts[name] = cnt
    st.session_state.single_type_results = single_results
    st.session_state.single_counts = single_counts

    # Mixed results
    if run_mixed:
        instances, total_units, placed_all = pack_mixed_optimizer(all_boxes, ULD_SPECS, VOLUME_UNITS)
        if instances:
            st.session_state.mixed = {"instances": instances, "total_units": total_units, "placed_all": placed_all}
        else:
            st.session_state.mixed = None
    else:
        st.session_state.mixed = None

# ==============================
# Visualization helpers
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
# Show results
# ==============================
if "single_type_results" in st.session_state or "mixed" in st.session_state:
    tab_single, tab_mixed = st.tabs(["Single ULD results", "Mixed ULD plan"])

    # --- Single ULD tab (2-row table + viz) ---
    with tab_single:
        counts = st.session_state.get("single_counts", {})
        if counts:
            # Build 2-row table: Type of ULD (row 1), Amount needed (row 2)
            used_types = [t for t in ULD_ORDER if t in counts]
            data = [
                [t for t in used_types],            # Row 1
                [counts[t] for t in used_types],    # Row 2
            ]
            df = pd.DataFrame(
                data,
                index=["Type of ULD", "Amount needed"],
                columns=used_types
            )
            st.subheader("Single-ULD results (FFD)")
            st.table(df)

            # Visualize a selected single-ULD packing
            choice = st.selectbox("Select ULD to visualize", used_types)
            chosen_result, cnt, spec = st.session_state.single_type_results[choice]
            st.info(f"{choice}: {cnt} used")

            gap = st.slider("ULD spacing (cm)", 200, 300, 250, 10, key="gap_single")
            OFFSET = gap

            fig = go.Figure()

            # ULD frames
            for uld_id in sorted(set(b["uld_id"] for b in chosen_result)):
                x0 = uld_id * OFFSET
                x1 = x0 + spec["length"]
                y1, z1 = spec["width"], spec["height"]
                corners = [[x0,0,0],[x1,0,0],[x1,y1,0],[x0,y1,0],
                           [x0,0,z1],[x1,0,z1],[x1,y1,z1],[x0,y1,z1]]
                draw_box_wireframe(fig, corners, color="#666666", width=3)

            # Boxes (standard color)
            for b in chosen_result:
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
            # Defensive rebuild of placed_all
            placed_all = mix.get("placed_all") or []
            if any("uld_idx" not in b for b in placed_all):
                rebuilt = []
                for idx, inst in enumerate(mix["instances"]):
                    for p in inst["packed"]:
                        q = dict(p)
                        q["uld_idx"] = idx
                        rebuilt.append(q)
                mix["placed_all"] = rebuilt
                st.session_state.mixed = mix

           
            type_counts = Counter(inst["type"] for inst in mix["instances"])
            used_types = [t for t in ULD_ORDER if type_counts[t] > 0]
            if not used_types:
                st.info("No ULDs used for the current inputs.")
            else:
                data = [
                    [t for t in used_types],
                    [type_counts[t] for t in used_types],
                ]
                df = pd.DataFrame(
                    data,
                    index=["Type of ULD", "Amount needed"],
                    columns=used_types
                )
                st.subheader("Mixed ULD plan")
                st.table(df)

            # Visualization
            gap = st.slider("ULD spacing (cm)", 200, 300, 250, 10, key="gap_mixed")
            OFFSET = gap

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

            # Boxes (standard color)
            for b in mix["placed_all"]:
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


