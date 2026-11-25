#!/usr/bin/env python3
"""Regenerate the GBZ test fixture (biopath src/gbz_input.rs tests):
   python3 make_small.py > small.gfa
   vg gbwt -G small.gfa --gbz-format --gbz-version 1 -g small.gbz   (the gbwt crate reads version 1)
   vg gbwt -G small.gfa --gbz-format -g small.v2.gbz                 (vg 1.77 default: version 2)
   vg convert -f --no-translation small.gbz | grep -v $'^H\\tNM:' > small.vg.gfa   (vg 1.77;
   biopath reads one H line only)
Two contigs; per step a SNP bubble, an inversion, a tandem self loop (z+ -> z+) or a plain step;
HG002#2 ends chr1 with a hairpin (y+ -> y-). Every edge is used by some walk (a GBZ keeps no other)."""
import random
rng = random.Random(11)
S = {}
nid = [0]
def node(lo=1, hi=12):
    nid[0] += 1
    S[nid[0]] = "".join(rng.choice("ACGT") for _ in range(rng.randint(lo, hi)))
    return nid[0]
haps = [("GRCh38", "0"), ("HG002", "1"), ("HG002", "2"), ("HG00099", "1")]
W = []
for chrom, steps in (("chr1", 90), ("chr2", 40)):
    walks = {h: [] for h in haps}
    prev = node()
    for h in haps: walks[h].append(">%d" % prev)
    for _ in range(steps):
        nxt = node(); r = rng.random()
        if r < 0.35:
            a, b = node(), node()
            for h in haps: walks[h].append(">%d" % rng.choice([a, b]))
        elif r < 0.5:
            v = node(20, 40)
            for h in haps: walks[h].append("<%d" % v if rng.random() < 0.5 else ">%d" % v)
        elif r < 0.58:
            z = node()
            for h in haps: walks[h].extend([">%d" % z] * rng.choice([1, 2, 3]))
        for h in haps: walks[h].append(">%d" % nxt)
        prev = nxt
    if chrom == "chr1":
        y = node()
        walks[("HG002", "2")].extend([">%d" % y, "<%d" % y])
    for (s, h), w in walks.items():
        ln = sum(len(S[int(x[1:])]) for x in w)
        W.append("W\t%s\t%s\t%s\t0\t%d\t%s" % (s, h, chrom, ln, "".join(w)))
links = set()
for line in W:
    steps = [(int(x[1:]), x[0] == ">") for x in __import__("re").findall(r"[<>]\d+", line.split("\t")[6])]
    for (a, af), (b, bf) in zip(steps, steps[1:]):
        links.add((a, "+" if af else "-", b, "+" if bf else "-"))
print("H\tVN:Z:1.1")
for n, s in S.items(): print("S\t%d\t%s" % (n, s))
for a, ao, b, bo in sorted(links): print("L\t%d\t%s\t%d\t%s\t0M" % (a, ao, b, bo))
for line in W: print(line)
