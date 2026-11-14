"""Make video-demo-prompter/: the same video plus the script lines (who + what to say) at the bottom, and a clock."""
import html, re, shutil, subprocess
from pathlib import Path
src, dst = Path("video-demo"), Path("video-demo-prompter")
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".hyperframes", "renders"))
rows = [l for l in (src / "SCRIPT.md").read_text().splitlines() if re.match(r"\| \d:\d\d \|", l)]
cues = []
for r in rows:
    t, who, say = [c.strip() for c in r.strip("|").split("|")[:3]]
    m, s = t.split(":"); cues.append((int(m) * 60 + int(s), who.strip("*"), say))
els, tw = [], []
for i, (start, who, say) in enumerate(cues):
    end = cues[i + 1][0] if i + 1 < len(cues) else 118
    col = {"Kabir": "#7FC4FF", "Vishnu": "#FFC46B", "Both": "#7FE0B8"}[who]
    words = say.split()
    # Karaoke pacing: words light up at a calm ~2.5 words/second, starting 0.2 s into the line,
    # and always finish before the next line (if the slot is tight, they speed up to fit).
    step = min(0.4, (end - start - 0.5) / max(len(words), 1))
    spans = " ".join(f'<i id="c{i}w{j}">{html.escape(w)}</i>' for j, w in enumerate(words))
    els.append(f'<div class="cue" id="cue{i}"><b style="color:{col}">{who.upper()}</b><span>{spans}</span></div>')
    tw.append(f'tl.set("#cue{i}", {{ opacity: 1 }}, {start}).set("#cue{i}", {{ opacity: 0 }}, {end - 0.05});')
    for j in range(len(words)):
        tw.append(f'tl.set("#c{i}w{j}", {{ color: "{col}" }}, {start + 0.2 + j * step:.2f});')
(dst / "compositions" / "cue.html").write_text(f'''<template>
  <style>
    #root {{ position: absolute; inset: 0; pointer-events: none; font-family: Inter, sans-serif; }}
    .band {{ position: absolute; left: 0; right: 0; bottom: 0; height: 250px; background: rgba(0,0,0,.86); }}
    .cue {{ position: absolute; left: 80px; right: 80px; bottom: 40px; height: 170px; color: #fff; font-size: 44px; line-height: 1.25; opacity: 0; display: flex; gap: 30px; align-items: center; }}
    .cue b {{ font-size: 40px; min-width: 200px; }}
    .cue i {{ font-style: normal; color: rgba(255,255,255,.55); }}
    .clock {{ position: absolute; right: 30px; top: 24px; color: #fff; background: rgba(0,0,0,.7); padding: 8px 16px; border-radius: 10px; font-size: 34px; font-family: "IBM Plex Mono", monospace; }}
  </style>
  <div id="root" data-composition-id="cue" data-width="1920" data-height="1080">
    <div class="band" data-layout-allow-overlap></div>
    {"".join(els)}
    <div class="clock" id="clock">0:00</div>
  </div>
  <script>
    const tl = gsap.timeline({{ paused: true }});
    {" ".join(tw)}
    const c = {{ t: 0 }};
    tl.to(c, {{ t: 118, duration: 118, ease: "none", onUpdate: () => {{ const s = Math.floor(c.t); document.getElementById("clock").textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); }} }}, 0);
    window.__timelines["cue"] = tl;
  </script>
</template>
''')
idx = dst / "index.html"; s = idx.read_text()
s = s.replace('    </div>\n    <script>', '      <div id="cue" data-composition-id="cue" data-composition-src="compositions/cue.html" data-start="0" data-duration="118" data-track-index="2" data-track-kind="graphics" data-width="1920" data-height="1080" style="z-index:10"></div>\n    </div>\n    <script>', 1)
idx.write_text(s)
print(len(cues), "cues")
