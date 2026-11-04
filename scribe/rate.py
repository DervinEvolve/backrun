"""A local page for rating pool candidates into love, okay and nope.

    python -m scribe.rate POOL_ROOT [--port 5427]

Open http://127.0.0.1:5427/ in a browser. One candidate at a time, at the
512 px preview, with its prompt. Keys: L love, O okay, N nope, U undo. The
generator model is not shown, so ratings stay blind to it. Every rating is
one line appended to ``POOL_ROOT/ratings.jsonl``. Standard library only; the
page is one HTML string with inline CSS and JavaScript.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from scribe.pool import TIERS, Pool


class RatingApp:
    """The rating logic, kept apart from HTTP so it can be tested directly."""

    def __init__(self, pool: Pool) -> None:
        self.pool = pool

    def _card(self, cid: str) -> dict[str, Any]:
        meta = self.pool.meta(cid)
        return {"id": cid, "subject": meta.get("subject", ""), "prompt": meta.get("prompt", ""),
                "image": f"/img/{cid}"}

    def state(self, focus: str | None = None) -> dict[str, Any]:
        """What the page shows: the current candidate (``focus`` or the next unrated) and progress."""
        s = self.pool.summary()
        current = None
        if focus is not None:
            current = self._card(focus)
        else:
            nxt = self.pool.next_unrated()
            if nxt is not None:
                current = self._card(nxt["id"])
        if current is not None:
            current["tier"] = self.pool.ratings().get(current["id"])
        return {"current": current, "rated": s["rated"], "total": s["total"],
                "counts": {t: s["tiers"][t] for t in TIERS}}

    def rate(self, cid: str, tier: str) -> dict[str, Any]:
        """Record a rating and return the next state."""
        self.pool.rate(cid, tier)
        return self.state()

    def undo(self) -> dict[str, Any]:
        """Undo the last rating and return a state focused on the candidate it restored."""
        cid = self.pool.undo()
        return self.state(focus=cid)

    def image_path(self, cid: str) -> Path:
        """The preview file for an indexed candidate; KeyError for anything else."""
        folder = self.pool.folder(cid)
        p = folder / "preview.png"
        return p if p.is_file() else folder / "final.png"


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pool Rating Desk</title>
<style>
:root{--desk:#6f5641;--desk2:#5d4634;--paper:#f6f1e6;--ink:#2d2620;--muted:#6c6157;--brass:#b59a5a;
--love:#6f8f4e;--okay:#b08a3e;--nope:#9a5148;--drawer:#8a6c52;--drawer2:#7a5e46}
@media (prefers-color-scheme: dark){:root{--desk:#3a2d23;--desk2:#2f241c;--paper:#e9e2d3;--drawer:#5c4635;--drawer2:#4e3b2d}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;font:16px/1.4 Georgia,"Iowan Old Style",serif;color:var(--ink);
background:repeating-linear-gradient(92deg,var(--desk) 0 14px,var(--desk2) 14px 15px,var(--desk) 15px 40px);
display:flex;flex-direction:column;align-items:center;padding:16px}
header{color:#f3ead9;width:min(560px,100%);display:flex;justify-content:space-between;align-items:baseline;gap:12px}
header h1{font-size:1.1rem;font-weight:normal;letter-spacing:.04em;margin:0}
#progress{font-variant-numeric:tabular-nums;font-size:.95rem}
.stage{position:relative;width:min(560px,100%);margin-top:12px;min-height:420px}
.sheet{background:var(--paper);padding:16px 16px 12px;border-radius:2px;width:100%;
box-shadow:0 1px 0 #0002,0 10px 24px #0005;transition:transform .45s cubic-bezier(.5,0,.75,0),opacity .45s;position:relative}
.sheet::before,.sheet::after{content:"";position:absolute;top:-8px;width:64px;height:20px;background:#efe6c8cc;
box-shadow:0 1px 2px #0002}
.sheet::before{left:24px;transform:rotate(-4deg)}.sheet::after{right:24px;transform:rotate(3deg)}
.sheet img{display:block;width:100%;height:auto;background:#fff;image-rendering:auto}
.sheet .subject{margin:10px 0 2px;font-style:italic}
.sheet .prompt{margin:0;color:var(--muted);font-size:.9rem}
.sheet.go-love{transform:translate(-36%,120%) scale(.4) rotate(-6deg);opacity:0}
.sheet.go-okay{transform:translate(0,125%) scale(.4);opacity:0}
.sheet.go-nope{transform:translate(36%,120%) scale(.4) rotate(6deg);opacity:0}
.sheet.enter{transform:translateY(-30px);opacity:0;transition:none}
.done{color:#f3ead9;text-align:center;padding:80px 0}
.cabinet{width:min(560px,100%);display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:18px;
background:var(--drawer2);padding:8px;border-radius:4px;box-shadow:inset 0 2px 6px #0006}
.drawer{background:linear-gradient(var(--drawer),var(--drawer2));border:1px solid #0003;border-radius:3px;
padding:10px 6px 8px;text-align:center;color:#f3ead9;cursor:pointer;font:inherit;transition:transform .15s}
.drawer:hover,.drawer:focus-visible{transform:translateY(2px)}
.drawer.bump{transform:translateY(5px)}
.plate{display:inline-block;background:linear-gradient(#d8c48c,var(--brass));color:#2d2620;
padding:2px 10px;border-radius:2px;font-size:.85rem;letter-spacing:.05em;box-shadow:0 1px 1px #0005}
.handle{display:block;width:44px;height:6px;margin:8px auto 4px;border-radius:3px;background:var(--brass);box-shadow:0 1px 1px #0006}
.count{font-variant-numeric:tabular-nums;font-size:.85rem}
.drawer[data-tier=love] .plate{border-left:4px solid var(--love)}
.drawer[data-tier=okay] .plate{border-left:4px solid var(--okay)}
.drawer[data-tier=nope] .plate{border-left:4px solid var(--nope)}
.keys{color:#e8dcc6;font-size:.85rem;margin-top:10px;text-align:center}
kbd{font:inherit;background:var(--paper);color:var(--ink);border-radius:2px;padding:0 5px;box-shadow:0 1px 0 #0006}
#msg{color:#ffd7cf;min-height:1.2em;font-size:.85rem}
</style></head><body>
<header><h1>Rating desk</h1><span id="progress">loading</span></header>
<div class="stage" id="stage"></div>
<div class="cabinet">
 <button class="drawer" data-tier="love"><span class="plate">LOVE</span><span class="handle"></span><span class="count" id="c-love">0</span></button>
 <button class="drawer" data-tier="okay"><span class="plate">OKAY</span><span class="handle"></span><span class="count" id="c-okay">0</span></button>
 <button class="drawer" data-tier="nope"><span class="plate">NOPE</span><span class="handle"></span><span class="count" id="c-nope">0</span></button>
</div>
<div class="keys"><kbd>L</kbd> love &nbsp; <kbd>O</kbd> okay &nbsp; <kbd>N</kbd> nope &nbsp; <kbd>U</kbd> undo</div>
<div id="msg"></div>
<script>
let state=null,busy=false;
const fmt=n=>n.toLocaleString("en-US");
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function draw(enter){
  document.getElementById("progress").textContent=`rated ${fmt(state.rated)} of ${fmt(state.total)}`;
  for(const t of ["love","okay","nope"])document.getElementById("c-"+t).textContent=fmt(state.counts[t]);
  const st=document.getElementById("stage"),c=state.current;
  if(!c){st.innerHTML='<div class="done">Every candidate is rated.</div>';return}
  const was=c.tier?` (was ${esc(c.tier)})`:"";
  st.innerHTML=`<div class="sheet${enter?" enter":""}" id="sheet"><img alt="${esc(c.subject)}" src="${esc(c.image)}">
  <p class="subject">${esc(c.subject)}${was}</p><p class="prompt">${esc(c.prompt)}</p></div>`;
  if(enter)requestAnimationFrame(()=>requestAnimationFrame(()=>{const s=document.getElementById("sheet");
    s.style.transition="transform .3s ease-out, opacity .3s";s.classList.remove("enter")}));
}
async function call(path,body){
  const r=await fetch(path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json"},body:body?JSON.stringify(body):undefined});
  if(!r.ok)throw new Error(await r.text());return r.json();
}
async function rate(tier){
  if(busy||!state||!state.current)return;busy=true;
  const sheet=document.getElementById("sheet"),drawer=document.querySelector(`.drawer[data-tier=${tier}]`);
  sheet.classList.add("go-"+tier);
  try{
    const next=await call("/api/rate",{id:state.current.id,tier});
    await new Promise(r=>setTimeout(r,420));drawer.classList.add("bump");setTimeout(()=>drawer.classList.remove("bump"),160);
    state=next;draw(true);document.getElementById("msg").textContent="";
  }catch(e){sheet.classList.remove("go-"+tier);document.getElementById("msg").textContent="Not saved: "+e.message}
  busy=false;
}
async function undo(){
  if(busy)return;busy=true;
  try{state=await call("/api/undo",{});draw(true)}catch(e){document.getElementById("msg").textContent=e.message}
  busy=false;
}
document.addEventListener("keydown",e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;const k=e.key.toLowerCase();
  if(k==="l")rate("love");else if(k==="o")rate("okay");else if(k==="n")rate("nope");else if(k==="u")undo();
});
document.querySelectorAll(".drawer").forEach(d=>d.addEventListener("click",()=>rate(d.dataset.tier)));
call("/api/state").then(s=>{state=s;draw(false)}).catch(e=>{document.getElementById("msg").textContent=e.message});
</script></body></html>
"""


def make_handler(app: RatingApp) -> type[BaseHTTPRequestHandler]:
    """An http.server handler class bound to one RatingApp."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(app.state())
            elif self.path.startswith("/img/"):
                try:
                    p = app.image_path(unquote(self.path[len("/img/"):]))
                    self._send(200, p.read_bytes(), "image/png")
                except (KeyError, OSError):
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/rate":
                    self._json(app.rate(str(body["id"]), str(body["tier"])))
                elif self.path == "/api/undo":
                    self._json(app.undo())
                else:
                    self._send(404, b"not found", "text/plain")
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._send(400, f"bad request: {exc}".encode(), "text/plain")

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def serve(pool_root: str | Path, port: int = 5427, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Build the server (not started). Call ``serve_forever()`` on the result."""
    return ThreadingHTTPServer((host, port), make_handler(RatingApp(Pool(pool_root))))


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    ap = argparse.ArgumentParser(description="Rate pool candidates into love, okay and nope.")
    ap.add_argument("pool_root")
    ap.add_argument("--port", type=int, default=5427)
    args = ap.parse_args(argv)
    srv = serve(args.pool_root, args.port)
    s = Pool(args.pool_root).summary()
    print(f"rating desk at http://127.0.0.1:{srv.server_address[1]}/  ({s['rated']} of {s['total']} rated)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
