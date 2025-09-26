import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from numpy.fft import rfft, rfftfreq, irfft

# 路径
FEATURES_CSV = r"data/2features/feat_source_32k_DE.csv"
OUT_DIR      = r"result/analysis"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 全局风格
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "grid.color": "#AAAAAA",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
})
CB_PALETTE = plt.get_cmap("tab10").colors
DEF_FIGSIZE = (6, 4)

def savefig(fig, path_wo_ext: str):
    png = path_wo_ext + ".png"
    fig.tight_layout(pad=0.5)
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 图表保存：{os.path.basename(png)}")

def style_axes(ax):
    ax.grid(True, axis="y", alpha=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)

# 读取特征表
df = pd.read_csv(FEATURES_CSV)
labels_order = sorted([str(x) for x in df["label"].dropna().unique()])

# 各类特征均值
cols_mean = ["rms","kurtosis","skewness","crest_factor",
             "spec_entropy","BPFO_ratio","BPFI_ratio","BSF_ratio"]
(df.groupby("label")[ [c for c in cols_mean if c in df.columns] ]
   .mean().reset_index()).to_csv(
       os.path.join(OUT_DIR,"feature_means_by_label.csv"),
       index=False, encoding="utf-8-sig")

# 箱线图
def save_boxplot(df, ycol, out_wo_ext):
    data = [df.loc[df["label"]==lab,ycol].dropna() for lab in labels_order]
    keep = [i for i,a in enumerate(data) if len(a)>0]
    if not keep: return
    tick_labels = [labels_order[i] for i in keep]
    data = [data[i] for i in keep]
    fig, ax = plt.subplots(figsize=DEF_FIGSIZE)
    bp = ax.boxplot(data, tick_labels=tick_labels, showfliers=False, widths=0.6,
                    patch_artist=True, medianprops=dict(color="black",lw=1.2))
    for i,p in enumerate(bp["boxes"]):
        p.set_facecolor(CB_PALETTE[i % len(CB_PALETTE)]); p.set_alpha(0.25)
    ax.set_xlabel("Label"); ax.set_ylabel(ycol); ax.set_title(f"{ycol} by Label")
    style_axes(ax); savefig(fig, out_wo_ext)

# 直方图
def save_hist(df, xcol, out_wo_ext, bins=30):
    fig, ax = plt.subplots(figsize=DEF_FIGSIZE); used=0
    for i,lab in enumerate(labels_order):
        x = df.loc[df["label"]==lab,xcol].dropna().values
        if len(x)==0: continue
        ax.hist(x,bins=bins,histtype="step",lw=1.2,
                label=lab,color=CB_PALETTE[i%len(CB_PALETTE)],alpha=0.95)
        used+=1
    if used==0: plt.close(fig); return
    ax.set_xlabel(xcol); ax.set_ylabel("Count"); ax.set_title(f"Distribution of {xcol}")
    style_axes(ax); ax.legend(frameon=False,ncol=min(3,used)); savefig(fig,out_wo_ext)

# 热力图
def save_corr_heatmap(df, cols, out_wo_ext):
    cols=[c for c in cols if c in df.columns]; sub=df[cols].dropna()
    if sub.empty or len(cols)<2: return
    corr=sub.corr(numeric_only=True).values
    fig, ax = plt.subplots(figsize=(0.6*len(cols)+3,0.6*len(cols)))
    im=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val=corr[i,j]; ax.text(j,i,f"{val:.2f}",ha="center",va="center",
                                   color="white" if abs(val)>0.5 else "black",fontsize=9)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=45,ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    ax.set_title("Feature Correlation Heatmap"); style_axes(ax)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04).ax.set_ylabel("Pearson r")
    savefig(fig,out_wo_ext)

# === 新增：时域/包络谱/倒谱 ===
def plot_time_snippet(sig, fs, out_wo_ext, n_points=2000):
    fig, ax = plt.subplots(figsize=DEF_FIGSIZE)
    t = np.arange(min(n_points,len(sig)))/fs
    ax.plot(t, sig[:len(t)], lw=0.8, color="k")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Amplitude")
    ax.set_title("Time-domain snippet")
    style_axes(ax); savefig(fig, out_wo_ext)

def plot_envelope_spectrum(sig, fs, out_wo_ext, freqs_mark=None):
    env = np.abs(hilbert(sig))
    N=len(env); F=rfftfreq(N,1/fs); A2=np.abs(rfft(env*np.hanning(N)))**2
    fig, ax = plt.subplots(figsize=DEF_FIGSIZE)
    ax.plot(F,10*np.log10(A2+1e-12),lw=0.8,color="k")
    if freqs_mark:
        for f in freqs_mark:
            ax.axvline(f,color="r",ls="--",lw=0.8)
    ax.set_xlim(0,fs/2); ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Envelope Spectrum [dB]")
    ax.set_title("Envelope Spectrum"); style_axes(ax); savefig(fig,out_wo_ext)

def plot_cepstrum(sig, fs, out_wo_ext, max_q=0.02):
    N=len(sig); F,A2=None,None
    X=np.abs(rfft(sig*np.hanning(N)))+1e-12
    logmag=np.log(X); sym=np.concatenate([logmag,logmag[-2:0:-1]])
    ceps=np.abs(irfft(sym)); q=np.arange(len(ceps))/fs
    fig, ax=plt.subplots(figsize=DEF_FIGSIZE)
    m=int(max_q*fs); ax.plot(q[:m],ceps[:m],lw=0.8,color="k")
    ax.set_xlabel("Quefrency [s]"); ax.set_ylabel("Cepstrum Amplitude")
    ax.set_title("Cepstrum"); style_axes(ax); savefig(fig,out_wo_ext)

# === 执行 ===
for c in ["rms","kurtosis","spec_entropy","BPFO_ratio","BPFI_ratio","BSF_ratio"]:
    if c in df.columns: save_boxplot(df,c,os.path.join(OUT_DIR,f"box_{c}"))
for c in ["rms","kurtosis","BPFO_ratio"]:
    if c in df.columns: save_hist(df,c,os.path.join(OUT_DIR,f"hist_{c}"))
save_corr_heatmap(df, cols_mean, os.path.join(OUT_DIR,"corr_heatmap"))

# === 示例：额外画几张典型图（需先读一段原始信号） ===
import scipy.io as sio
file_path="../data/0数据集/源域数据集/12kHz_DE_data/B/0014/B014_1.mat"
data = sio.loadmat(file_path)
last_key = ([k for k in data.keys() if not k.startswith("__")])[-4]
sig = data[last_key].squeeze()
fs = 12000
plot_time_snippet(sig, fs, os.path.join(OUT_DIR,"time_example"))
plot_envelope_spectrum(sig, fs, os.path.join(OUT_DIR,"envspec_example"), freqs_mark=[100,200])
plot_cepstrum(sig, fs, os.path.join(OUT_DIR,"cepstrum_example"))
