# Chen-G.-Cao-B.-He-X.-2025-.-Title-of-dataset-v1.0-
import pandas as pd
import numpy as np
import os, json

RAW_DIR = os.path.dirname(__file__)

def iou_rect(a, b):
    # a, b: (cx, cy, w, h)
    ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2
    ax2, ay2 = a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2
    bx2, by2 = b[0]+b[2]/2, b[1]+b[3]/2
    inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
    inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, inter_x2-inter_x1), max(0.0, inter_y2-inter_y1)
    inter = iw*ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union>0 else 0.0

def eval_map(det_file, gt_file, iou_thr=0.5):
    det = pd.read_csv(os.path.join(RAW_DIR, det_file))
    gt  = pd.read_csv(os.path.join(RAW_DIR, gt_file))
    aps = []
    # Evaluate per time-slice then average (pedagogical simplification)
    for t, gts in gt.groupby("time_s"):
        preds = det[det["time_s"]==t].sort_values("confidence", ascending=False)
        # consider only non-spurious or include FP; FPs remain unmatched
        gt_boxes = [(r.gt_cx_m, r.gt_cy_m, r.gt_w_m, r.gt_h_m, r.obj_id) for _, r in gts.iterrows()]
        matched = set()
        tp, fp = [], []
        for _, p in preds.iterrows():
            pb = (p.pred_cx_m, p.pred_cy_m, p.pred_w_m, p.pred_h_m)
            best_iou, best_idx = 0.0, None
            for j, gb in enumerate(gt_boxes):
                if j in matched: continue
                iou = iou_rect(pb, gb)
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_iou >= iou_thr:
                matched.add(best_idx)
                tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)
        fn = len(gt_boxes) - len(matched)
        # Precision-Recall (11-pt like)
        if len(tp)+len(fp)==0: 
            aps.append(0.0); 
            continue
        prec = []
        rec = []
        cum_tp, cum_fp = 0, 0
        for i in range(len(tp)):
            cum_tp += tp[i]; cum_fp += fp[i]
            prec.append(cum_tp / max(1, (cum_tp+cum_fp)))
            rec.append(cum_tp / max(1, (cum_tp+fn)))
        # Simple AP approx: area under P-R step curve
        ap = 0.0
        last_r = 0.0
        for p, r in zip(prec, rec):
            ap += p * max(0, r - last_r)
            last_r = r
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0

def eval_ate_rte(loc_file, ego_file):
    loc = pd.read_csv(os.path.join(RAW_DIR, loc_file))
    ego = pd.read_csv(os.path.join(RAW_DIR, ego_file))
    df = pd.merge(loc, ego, on="time_s")
    ate = np.mean(np.sqrt((df["pose_x_m_est"]-df["ego_x_m"])**2 + (df["pose_y_m_est"]-df["ego_y_m"])**2))
    rte = np.mean(np.sqrt(np.diff(df["pose_x_m_est"])**2 + np.diff(df["pose_y_m_est"])**2))
    return float(ate), float(rte)

def eval_latency_p95(ctl_file):
    ctl = pd.read_csv(os.path.join(RAW_DIR, ctl_file))
    return float(np.percentile(ctl["measured_latency_ms"], 95))

def main():
    res = {}
    for tag in ["baseline","proposed"]:
        m = {}
        m["mAP"] = eval_map(f"detections_{tag}.csv", "scenario_ground_truth.csv", iou_thr=0.5)
        ate, rte = eval_ate_rte(f"localization_{tag}.csv", "ego_ground_truth.csv")
        m["ATE_m"], m["RTE_m"] = ate, rte
        m["latency_p95_ms"] = eval_latency_p95(f"control_loop_{tag}.csv")
        res[tag] = m
    out = pd.DataFrame(res).T
    out.to_csv(os.path.join(RAW_DIR, "metrics_from_raw.csv"))
    print(out)

if __name__ == "__main__":
    main()
