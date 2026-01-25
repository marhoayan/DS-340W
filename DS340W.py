#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/storage/home/oqa5143/work/DS340W/AA.csv")

# normalize column names
df.columns = [c.strip() for c in df.columns]

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sentiment_gpt"] = pd.to_numeric(df["Sentiment_gpt"], errors="coerce")

df = df.dropna(subset=["Date", "Sentiment_gpt"]).sort_values("Date").reset_index(drop=True)

df.head()


# FIGURE 4:

# In[2]:


OUT_DIR = "outputs"
SPLIT_DIR = os.path.join(OUT_DIR, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

n = len(df)
train_end = int(0.70 * n)
test_end  = int(0.90 * n)

train_df = df.iloc[:train_end].copy()
test_df  = df.iloc[train_end:test_end].copy()
val_df   = df.iloc[test_end:].copy()

train_df.to_csv(os.path.join(SPLIT_DIR, "AA_train.csv"), index=False)
test_df.to_csv(os.path.join(SPLIT_DIR, "AA_test.csv"), index=False)
val_df.to_csv(os.path.join(SPLIT_DIR, "AA_val.csv"), index=False)

print("Rows:", {"train": len(train_df), "test": len(test_df), "val": len(val_df)})
print("Saved to:", SPLIT_DIR)


# In[3]:


counts = df["Sentiment_gpt"].value_counts().sort_index()
percentages = counts / counts.sum() * 100

plt.figure()
bars = plt.bar(counts.index, counts.values)
plt.xticks([1,2,3,4,5])
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.title("GPT Sentiment Distribution")

for i, p in enumerate(percentages):
    plt.text(counts.index[i], counts.values[i], f"{p:.2f}%", ha='center', va='bottom')

plt.show()


# The reproduced sentiment distribution exhibits the same unimodal structure reported in the original paper, with the highest mass centered at the neutral score (3) and decreasing frequencies toward the negative (1) and positive (5) extremes, confirming the stability of the GPT-based sentiment labeling process.

# FIGURE 6:

# In[4]:


df["Year"] = df["Date"].dt.year
year_counts = df.groupby("Year").size()

x = year_counts.index.to_numpy()
y = year_counts.values

plt.figure(figsize=(8,5))
plt.bar(x, y)
plt.yscale("log")
plt.xlabel("Year")
plt.ylabel("News Number (log scale)")
plt.title("Yearly vs News Number")
plt.show()


# Figure X reproduces the yearly distribution of news volume on a logarithmic scale as in Figure 6 of the paper. While the absolute counts differ due to using a subset of the full FNSPID corpus, the temporal growth pattern and log-scale behavior are consistent with the original study.

# In[5]:


import os
path = "/storage/home/oqa5143/work/DS340W/nasdaq_exteral_data.csv"
print("Exists:", os.path.exists(path))
print("Size (GB):", os.path.getsize(path)/1024**3)


# In[6]:


with open("nasdaq_exteral_data.csv", "r", encoding="utf-8", errors="ignore") as f:
    header = f.readline().strip()
header[:300]


# FIGURE 5:

# In[7]:


import csv
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

file = "nasdaq_exteral_data.csv"

# These are "candidate" column names the dataset might use.
# We'll auto-detect which ones exist.
SYMBOL_COL_CANDIDATES = ["Symbol", "symbol", "Stock_symbol", "stock_symbol", "ticker", "Ticker"]
LANG_COL_CANDIDATES   = ["Language", "language", "lang"]
URL_COL_CANDIDATES    = ["Url", "URL", "url", "link"]
TEXT_COL_CANDIDATES   = ["Text", "text", "Article", "article", "content", "Content"]

def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

total = 0

symbol_present = 0
symbol_missing = 0

url_present = 0
url_missing = 0

text_present = 0
text_missing = 0

lang_counts = Counter()   # we'll focus on English/Russian if present

with open(file, "r", encoding="utf-8", errors="ignore", newline="") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames

    symbol_col = first_existing(cols, SYMBOL_COL_CANDIDATES)
    lang_col   = first_existing(cols, LANG_COL_CANDIDATES)
    url_col    = first_existing(cols, URL_COL_CANDIDATES)
    text_col   = first_existing(cols, TEXT_COL_CANDIDATES)

    print("Detected columns:")
    print(" symbol_col =", symbol_col)
    print(" lang_col   =", lang_col)
    print(" url_col    =", url_col)
    print(" text_col   =", text_col)

    for row in reader:
        total += 1

        # Symbol
        if symbol_col is not None:
            v = (row.get(symbol_col) or "").strip()
            if v:
                symbol_present += 1
            else:
                symbol_missing += 1

        # URL
        if url_col is not None:
            v = (row.get(url_col) or "").strip()
            if v:
                url_present += 1
            else:
                url_missing += 1

        # Text/Article
        if text_col is not None:
            v = (row.get(text_col) or "").strip()
            if v:
                text_present += 1
            else:
                text_missing += 1

        # Language (paper compares English vs Russian)
        if lang_col is not None:
            v = (row.get(lang_col) or "").strip().lower()
            if v:
                # normalize common variants
                if v in ["en", "eng", "english"]:
                    lang_counts["English"] += 1
                elif v in ["ru", "rus", "russian"]:
                    lang_counts["Russian"] += 1
                else:
                    lang_counts["Other"] += 1

        # progress ping (every 1 million rows)
        if total % 1_000_000 == 0:
            print(f"Processed {total:,} rows...")

print("\nDone.")
print("Total rows:", total)
print("Symbol present/missing:", symbol_present, symbol_missing)
print("URL present/missing:", url_present, url_missing)
print("Text present/missing:", text_present, text_missing)
print("Language counts:", dict(lang_counts))


# In[8]:


import matplotlib.pyplot as plt
import numpy as np

def pct(a, b):
    s = a + b
    return 0 if s == 0 else (100*a/s)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# A: Symbol presence
if symbol_present + symbol_missing > 0:
    vals = [symbol_missing, symbol_present]
    labels = ["Stock without Symbol", "Stock with Symbol"]
    axs[0,0].bar(labels, vals)
    axs[0,0].set_title("A")
    axs[0,0].set_ylim(0, max(vals)*1.15)
    axs[0,0].text(0, vals[0], f"{pct(symbol_missing, symbol_present):.2f}%", ha="center", va="bottom")
    axs[0,0].text(1, vals[1], f"{pct(symbol_present, symbol_missing):.2f}%", ha="center", va="bottom")
else:
    axs[0,0].set_title("A (no symbol column found)")
    axs[0,0].axis("off")

# B: Language distribution (English vs Russian)
if sum(lang_counts.values()) > 0:
    eng = lang_counts.get("English", 0)
    ru  = lang_counts.get("Russian", 0)
    vals = [eng, ru]
    labels = ["News in English", "News in Russian"]
    axs[0,1].bar(labels, vals)
    axs[0,1].set_title("B")
    axs[0,1].set_ylim(0, max(vals)*1.15 if max(vals)>0 else 1)
    total_lang = eng + ru
    if total_lang > 0:
        axs[0,1].text(0, vals[0], f"{(100*eng/total_lang):.2f}%", ha="center", va="bottom")
        axs[0,1].text(1, vals[1], f"{(100*ru/total_lang):.2f}%", ha="center", va="bottom")
else:
    axs[0,1].set_title("B (no language column found)")
    axs[0,1].axis("off")

# C: URL presence
vals = [url_missing, url_present]
labels = ["News without URL", "News with URL"]
axs[1,0].bar(labels, vals)
axs[1,0].set_title("C")
axs[1,0].set_ylim(0, max(vals)*1.15)
axs[1,0].text(0, vals[0], f"{pct(url_missing, url_present):.2f}%", ha="center", va="bottom")
axs[1,0].text(1, vals[1], f"{pct(url_present, url_missing):.2f}%", ha="center", va="bottom")

# D: Article/Text presence
if text_present + text_missing > 0:
    vals = [text_missing, text_present]
    labels = ["News without Article", "News with Article"]
    axs[1,1].bar(labels, vals)
    axs[1,1].set_title("D")
    axs[1,1].set_ylim(0, max(vals)*1.15)
    axs[1,1].text(0, vals[0], f"{pct(text_missing, text_present):.2f}%", ha="center", va="bottom")
    axs[1,1].text(1, vals[1], f"{pct(text_present, text_missing):.2f}%", ha="center", va="bottom")
else:
    axs[1,1].set_title("D (no text/article column found)")
    axs[1,1].axis("off")

plt.tight_layout()
plt.show()


# In[ ]:




