import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from collections import Counter, defaultdict
from functools import lru_cache

# ==============================
# ULD definitions
# ==============================
ULD_SPECS = {
    "AKE": {"length": 142, "width": 139, "height": 160, "max_weight": 1587},
    "LD7": {"length": 310, "width": 243, "height": 162, "max_weight": 5035},
    "PLA": {"length": 307, "width": 142, "height": 162, "max_weight": 3174},
}
VOLUME_UNITS = {"AKE": 4.0, "LD7": 10.0, "PLA": 8.0}
ULD_ORDER = ["AKE", "LD7", "PLA"]
MIXED_COLORS = {"AKE": "#000000", "LD7": "#00aa00", "PLA": "#cc0000"}
BOX_COLOR = "#1f77b4"

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Estimation Load Plan")

# ==============================
# Sidebar
# ==============================
st.sidebar.header("Settings")
run_mixed = st.sidebar.checkbox("Enable Mixed ULD mode", value=True)
default_spacing_cm = st.sidebar.slider("Default ULD spacing (cm)", 200, 800, 300, 10)
MAX_BOXES_TO_DRAW = st.sidebar.number_input("Max boxes to draw in 3D", 50, 5000, 400, 50)

# ==============================
# Input form
# ==============================
num_items = st.number_input("Number of cargo types", 1, 20, 1)
with st.form("cargo_input"):
    cargo_data = []
    for i in range(num_items):
        st.markdown(f"**Cargo {i+1}**")
        l = st.number_input(f"Length (cm) [{i+1}]", 1, 500, 100, key=f"l{i}")
        w = st.number_input(f"Width (cm)  [{i+1}]", 1, 500, 100, key=f"w{i}")
        h = st.number_input(f"Height (cm) [{i+1}]", 1, 500, 100, key=f"h{i}")
        wt = st.number_input(f"Weight (kg) [{i+1}]", 1, 1000, 100, key=f"wt{i}")
        q = st.number_input(f"Quantity [{i+1}]", 1, 200, 1, key=f"q{i}")
        cargo_data.append({"length": l, "width": w, "height": h, "weight": wt, "quantity": q})
    submitted = st.form_submit_button("Simulate")

# ==============================
# Helpers
# ==============================
def normalize_boxes(boxes):
    out = []
    for b in boxes:
        l, w = b["length"], b["width"]
        if w > l:
            l, w = w, l
        out.append({**b, "length": l, "width": w})
    return out


def pack_boxes_ffd_shelf(boxes, spec):
    L, W, H, WMAX = spec["length"], spec["width"], spec["height"], spec["max_weight"]
    boxes = sorted(boxes, key=lambda b: (b["length"]*b["width"], b["height"]), reverse=True)
    packed, uld_id, remaining = [], 0, boxes
    while remaining:
        used, placed, layer_z = 0, set(), 0
        any_placed = False
        while layer_z < H:
            row_y = 0; layer_h = 0; layer_ok = False
            while row_y < W:
                x = 0; row_d = 0; row_h = 0; row_ok = False
                for i,b in enumerate(remaining):
                    if i in placed: continue
                    l0,w0=b["length"],b["width"]
                    for bl,bw in ((l0,w0),(w0,l0)):
                        if bl<=L-x and bw<=W-row_y and b["height"]<=H-layer_z and used+b["weight"]<=WMAX:
                            packed.append({**b,"x":x,"y":row_y,"z":layer_z,"uld_id":uld_id})
                            placed.add(i)
                            x+=bl; row_d=max(row_d,bw); row_h=max(row_h,b["height"])
                            layer_h=max(layer_h,row_h); used+=b["weight"]
                            any_placed=True; row_ok=True; layer_ok=True; break
                    if row_ok: continue
                if not row_ok: break
                row_y+=row_d
            if not layer_ok: break
            layer_z+=max(1,layer_h)
        if not any_placed: return None,float("inf")
        remaining=[b for i,b in enumerate(remaining) if i not in placed]; uld_id+=1
    return packed,uld_id


def can_fit_all_in_one_instance(boxes, spec):
    """Quick check if all boxes fit in ONE instance"""
    L, W, H, WMAX = spec["length"], spec["width"], spec["height"], spec["max_weight"]
    todo = sorted(boxes, key=lambda b:(b["length"]*b["width"],b["height"]), reverse=True)
    used, layer_z = 0, 0
    while layer_z < H and todo:
        row_y = 0; layer_h = 0; lay_ok=False
        while row_y < W and todo:
            x = 0; row_d = 0; row_h = 0; row_ok=False; i=0
            while i < len(todo):
                b=todo[i]; l0,w0=b["length"],b["width"]
                for bl,bw in ((l0,w0),(w0,l0)):
                    if bl<=L-x and bw<=W-row_y and b["height"]<=H-layer_z and used+b["weight"]<=WMAX:
                        used+=b["weight"]; x+=bl
                        row_d=max(row_d,bw); row_h=max(row_h,b["height"])
                        layer_h=max(layer_h,row_h); todo.pop(i)
                        row_ok=True; lay_ok=True; break
                if not row_ok: i+=1
            if not row_ok: break
            row_y+=row_d
        if not lay_ok: break
        layer_z+=max(1,layer_h)
    return len(todo)==0


def boxes_key_tuple(boxes):
    return tuple(sorted((b["length"],b["width"],b["height"],b["weight"]) for b in boxes))

@lru_cache(maxsize=4096)
def _pack_ffd_cached(name,key):
    spec=ULD_SPECS[name]
    b=[{"length":l,"width":w,"height":h,"weight":wt} for l,w,h,wt in key]
    return pack_boxes_ffd_shelf(b,spec)


# ==============================
# Mixed optimizer with early guard
# ==============================
def pack_mixed_optimizer_fast(boxes, specs, units):
    remaining=list(boxes); plan=[]; total_units=0
    while remaining:
        key=boxes_key_tuple(remaining)

        # Early guard: if any single ULD fits all boxes -> choose min units
        single_fit=[]
        for name,spec in specs.items():
            if can_fit_all_in_one_instance(remaining,spec):
                single_fit.append((units[name],name,spec))
        if single_fit:
            single_fit.sort()
            _,nm,spec=single_fit[0]
            packed,_=_pack_ffd_cached(nm,key)
            grp=[p for p in packed if p["uld_id"]==0]
            for g in grp:g["uld_id"]=0
            plan.append({"type":nm,"spec":spec,"packed":grp})
            total_units+=units[nm]
            # remove
            tk=defaultdict(int)
            for p in grp: L=max(p["length"],p["width"]);W=min(p["length"],p["width"])
            tk[(L,W,p["height"],p["weight"])]+=1
            new=[]
            for b in remaining:
                k=(b["length"],b["width"],b["height"],b["weight"])
                if tk.get(k,0)>0: tk[k]-=1
                else:new.append(b)
            remaining=new; continue

        # Normal greedy
        best=None
        for name,spec in specs.items():
            pk,cnt=_pack_ffd_cached(name,key)
            if not pk or cnt==float("inf"): continue
            g=defaultdict(list)
            for p in pk:g[p["uld_id"]].append({**p})
            if 0 not in g: continue
            eff=len(g[0])/units[name]
            cand=(eff,-len(g[0]),name,spec,g[0])
            if not best or cand>best: best=cand
        if not best: break
        eff,_,nm,spec,grp=best
        for g in grp:g["uld_id"]=0
        plan.append({"type":nm,"spec":spec,"packed":grp})
        total_units+=units[nm]
        # remove grp
        tk=defaultdict(int)
        for p in grp: L=max(p["length"],p["width"]);W=min(p["length"],p["width"])
        tk[(L,W,p["height"],p["weight"])]+=1
        new=[]
        for b in remaining:
            k=(b["length"],b["width"],b["height"],b["weight"])
            if tk.get(k,0)>0: tk[k]-=1
            else:new.append(b)
        if len(new)==len(remaining): break
        remaining=new
    # flatten
    placed=[]
    for i,inst in enumerate(plan):
        for p in inst["packed"]:
            q=dict(p);q["uld_idx"]=i;placed.append(q)
    return plan,total_units,placed


# ==============================
# Simulation
# ==============================
if submitted:
    boxes=[]
    for c in cargo_data:
        for _ in range(c["quantity"]):
            boxes.append({"length":c["length"],"width":c["width"],"height":c["height"],"weight":c["weight"]})
    boxes=normalize_boxes(boxes)
    # single
    key=boxes_key_tuple(boxes)
    single_results,counts={},{}
    for n,s in ULD_SPECS.items():
        p,c=_pack_ffd_cached(n,key)
        if p: single_results[n]=(p,c,s); counts[n]=c
    st.session_state.single_type_results=single_results
    st.session_state.single_counts=counts
    # mixed
    if run_mixed:
        inst,units,placed=pack_mixed_optimizer_fast(boxes,ULD_SPECS,VOLUME_UNITS)
        st.session_state.mixed={"instances":inst,"total_units":units,"placed_all":placed} if inst else None
    else:
        st.session_state.mixed=None

# ==============================
# Visualisation
# ==============================
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
def draw_box_wireframe(fig,corners,color,width=2,showlegend=False,name=None):
    for e in EDGES:
        fig.add_trace(go.Scatter3d(x=[corners[e[0]][0],corners[e[1]][0]],
            y=[corners[e[0]][1],corners[e[1]][1]],
            z=[corners[e[0]][2],corners[e[1]][2]],
            mode="lines",line=dict(color=color,width=width),
            showlegend=showlegend,name=name))

if "single_type_results" in st.session_state or "mixed" in st.session_state:
    tab1,tab2=st.tabs(["Single ULD","Mixed ULD"])
    with tab1:
        if st.session_state.get("single_counts"):
            used=[t for t in ULD_ORDER if t in st.session_state.single_counts]
            df=pd.DataFrame([[t for t in used],
                             [st.session_state.single_counts[t] for t in used]],
                            index=["Type","Qty"],columns=used).astype(str)
            st.subheader("Single-ULD Results (FFD)"); st.table(df)
            ch=st.selectbox("Select ULD",used)
            pk,cnt,spec=st.session_state.single_type_results[ch]
            if st.button("Show 3D (Single)"):
                fig=go.Figure()
                gap=st.slider("Spacing",300,800,default_spacing_cm,10)
                for b in pk:
                    x=b["x"]+b["uld_id"]*gap; y,z=b["y"],b["z"]
                    l,w,h=b["length"],b["width"],b["height"]
                    c=[[x,y,z],[x+l,y,z],[x+l,y+w,z],[x,y+w,z],
                       [x,y,z+h],[x+l,y,z+h],[x+l,y+w,z+h],[x,y+w,z+h]]
                    draw_box_wireframe(fig,c,BOX_COLOR)
                st.plotly_chart(fig,use_container_width=True)
    with tab2:
        mix=st.session_state.get("mixed")
        if mix:
            tcnt=Counter(i["type"] for i in mix["instances"])
            used=[t for t in ULD_ORDER if tcnt[t]>0]
            df=pd.DataFrame([[t for t in used],[tcnt[t] for t in used]],
                            index=["Type","Qty"],columns=used)
            st.subheader("Mixed-ULD Plan (Min total units)"); st.table(df)
            if st.button("Show 3D (Mixed)"):
                fig=go.Figure(); gap=st.slider("Spacing",300,800,default_spacing_cm,10)
                for idx,inst in enumerate(mix["instances"]):
                    s=inst["spec"]; col=MIXED_COLORS[inst["type"]]
                    x0=idx*gap; x1=x0+s["length"]; y1,z1=s["width"],s["height"]
                    c=[[x0,0,0],[x1,0,0],[x1,y1,0],[x0,y1,0],
                       [x0,0,z1],[x1,0,z1],[x1,y1,z1],[x0,y1,z1]]
                    draw_box_wireframe(fig,c,col,4,True,f"{inst['type']}#{idx}")
                for b in mix["placed_all"]:
                    x=b["x"]+b["uld_idx"]*gap; y,z=b["y"],b["z"]
                    l,w,h=b["length"],b["width"],b["height"]
                    c=[[x,y,z],[x+l,y,z],[x+l,y+w,z],[x,y+w,z],
                       [x,y,z+h],[x+l,y,z+h],[x+l,y+w,z+h],[x,y+w,z+h]]
                    draw_box_wireframe(fig,c,BOX_COLOR)
                st.plotly_chart(fig,use_container_width=True)
