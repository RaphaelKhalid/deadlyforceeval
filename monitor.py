"""Live transcript monitor for run_or.py / run_anthropic.py runs. Stdlib only.

    python monitor.py [--port 8770]   ->  http://localhost:8770

Reads out/*/trials.jsonl on every poll, so new trials appear as they land. Cards sit in a
grid (1-4 columns or auto), every log section can be expanded at once, and each card shows
the exact 224x224 frame the model was sent.
"""
import glob
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "images")
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8770


def runs():
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, "out", "*", "trials.jsonl")), key=os.path.getmtime):  # newest last = default
        rows = []
        with open(path, encoding="utf-8") as f:
            for l in f:
                try:
                    rows.append(json.loads(l))
                except json.JSONDecodeError:
                    pass  # line still being written by the runner
        out[os.path.basename(os.path.dirname(path))] = rows
    return out


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Deadly-force monitor</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--ink:#e6e6e6;--mut:#8a93a3;--line:#2a2f3a;--refuse:#3fb27f;--fire:#e5484d;--aim:#f5a524;--put_down:#4ea1ff;--other:#8a93a3}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:10px 14px;align-items:center;flex-wrap:wrap}
select,button{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:4px 8px;font:inherit;font-size:13px;cursor:pointer}
button:hover{border-color:var(--mut)}button.on{border-color:var(--aim);color:var(--aim)}
.grp{display:flex;gap:4px;align-items:center}.lbl{color:var(--mut);font-size:12px}
.tally span{margin-right:10px;white-space:nowrap}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
main{padding:12px 16px;display:grid;gap:10px;align-items:start}
.t{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--other);border-radius:6px;padding:10px 12px;min-width:0}
.t h3{margin:0 0 6px;font-size:13.5px;display:flex;gap:8px;flex-wrap:wrap;align-items:baseline}.lab{font-weight:700;text-transform:uppercase}
.mut{color:var(--mut);font-weight:400;font-size:12px}pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;font:12px/1.4 ui-monospace,monospace}
details{margin-top:4px}details summary{cursor:pointer;color:var(--mut);font-size:12px}
.call{background:#0b0d11;padding:6px 8px;border-radius:4px}.row{display:flex;gap:10px;align-items:flex-start}
.row img{width:112px;height:112px;image-rendering:pixelated;border:1px solid var(--line);border-radius:3px;flex:none;cursor:zoom-in}
.row .body{min-width:0;flex:1}
#zoom{position:fixed;inset:0;background:#000c;display:none;align-items:center;justify-content:center;z-index:5;cursor:zoom-out}
#zoom img{width:min(90vmin,672px);height:min(90vmin,672px);image-rendering:pixelated}
</style></head><body>
<header><b>Deadly-force monitor</b>
<select id="run"></select><select id="cell"><option value="">all cells</option></select><select id="label"><option value="">all labels</option></select>
<span class="grp"><span class="lbl">columns</span><select id="cols"><option value="auto">auto</option><option>1</option><option>2</option><option>3</option><option>4</option></select></span>
<span class="grp"><span class="lbl">expand</span><button data-x="reasoning">reasoning</button><button data-x="content">text</button><button data-x="call">tool call</button><button id="all">all</button><button id="none">none</button></span>
<button id="img" class="on">images</button><select id="sort"><option value="new">newest first</option><option value="id">by trial id</option></select>
<span class="tally" id="tally"></span><span class="mut" id="meta"></span></header>
<main id="list"></main><div id="zoom"><img></div>
<script>
const C={refuse:'--refuse',fire:'--fire',aim:'--aim',put_down:'--put_down'};
const col=l=>`var(${C[l]||'--other'})`, esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const $=id=>document.getElementById(id);
let S={open:{reasoning:false,content:true,call:true},img:true}, manual=new Map(), sig='';
try{Object.assign(S,JSON.parse(localStorage.getItem('dfm')||'{}'))}catch(e){}
const save=()=>{try{localStorage.setItem('dfm',JSON.stringify({open:S.open,img:S.img,cols:$('cols').value,sort:$('sort').value}))}catch(e){}};
try{const s=JSON.parse(localStorage.getItem('dfm')||'{}');if(s.cols)$('cols').value=s.cols;if(s.sort)$('sort').value=s.sort}catch(e){}
function syncButtons(){document.querySelectorAll('[data-x]').forEach(b=>b.classList.toggle('on',S.open[b.dataset.x]));$('img').classList.toggle('on',S.img)}
function layout(){const c=$('cols').value;$('list').style.gridTemplateColumns=c==='auto'?'repeat(auto-fill,minmax(440px,1fr))':`repeat(${c},minmax(0,1fr))`}
function isOpen(id,kind){const k=id+'|'+kind;return manual.has(k)?manual.get(k):S.open[kind]}
function sec(id,kind,title,html){return html?`<details data-k="${esc(id)}|${kind}" ${isOpen(id,kind)?'open':''}><summary>${title}</summary>${html}</details>`:''}
async function tick(force){
  let d; try{d=await (await fetch('/data')).json()}catch(e){return}
  const rs=$('run'),cs=$('cell'),ls=$('label'),names=Object.keys(d);
  if(rs.options.length!==names.length){const cur=rs.value;rs.innerHTML=names.map(n=>`<option>${n}</option>`).join('');rs.value=names.includes(cur)?cur:names[names.length-1]}
  const all=d[rs.value]||[], cells=[...new Set(all.map(r=>r.cell))], labs=[...new Set(all.map(r=>r.label||'error'))];
  const fill=(el,vals,first)=>{const k=vals.join();if(el.dataset.k!==k){const cur=el.value;el.innerHTML=`<option value="">${first}</option>`+vals.map(v=>`<option>${v}</option>`).join('');el.value=vals.includes(cur)?cur:'';el.dataset.k=k}};
  fill(cs,cells,'all cells');fill(ls,labs,'all labels');
  let R=all.filter(r=>(!cs.value||r.cell===cs.value)&&(!ls.value||(r.label||'error')===ls.value));
  R=$('sort').value==='id'?R.sort((a,b)=>a.trial_id.localeCompare(b.trial_id)):R.sort((a,b)=>b.ts-a.ts);
  const n={};R.forEach(r=>n[r.label||'error']=(n[r.label||'error']||0)+1);
  const ok=R.filter(r=>!r.error), ref=ok.filter(r=>r.refused).length, dec=ok.filter(r=>r.label!=='truncated').length;
  $('tally').innerHTML=Object.entries(n).map(([k,v])=>`<span><i class="dot" style="background:${col(k)}"></i>${k} ${v}</span>`).join('')+`<b>refused ${ref}/${dec}</b>`;
  const cost=ok.reduce((s,r)=>s+((r.usage||{}).cost||0),0);
  $('meta').textContent=`${R.length} trials · $${cost.toFixed(4)} · ${R[0]?R[0].model:''} · ${R[0]?(R[0].system_prompt||''):''}`;
  const ns=JSON.stringify([rs.value,cs.value,ls.value,$('sort').value,R.length,R.map(r=>r.ts).join(),S]);
  if(ns===sig&&!force)return; sig=ns;
  $('list').innerHTML=R.map(r=>{
    const tc=((r.tool_calls||[]).slice(-1)[0]||{}).function||{}, lab=r.label||'error', id=r.trial_id;
    let args=tc.arguments||''; try{args=JSON.stringify(JSON.parse(args),null,1)}catch(e){}
    const img=S.img&&r.image?`<img src="/${esc(r.image)}" title="${esc(r.image)}" alt="">`:'';
    return `<div class="t" style="border-left-color:${col(lab)}"><h3><span class="lab" style="color:${col(lab)}">${lab}</span>${esc(id)}
      <span class="mut">${esc(r.instruction)} · ${esc(r.provider)} · ${(r.usage||{}).completion_tokens??''} out tok · ${esc(r.finish_reason||'')} · ${new Date(r.ts*1000).toLocaleTimeString()}</span></h3>
      <div class="row">${img}<div class="body">
      ${r.error?`<pre>${esc(r.error)}</pre>`:''}
      ${sec(id,'call',`tool call: <b>${esc(tc.name||'(none)')}</b>`,`<div class="call"><pre>${esc(args)}</pre></div>`)}
      ${sec(id,'content','text',r.content?`<pre>${esc(r.content)}</pre>`:'')}
      ${sec(id,'reasoning',`reasoning (${(r.reasoning||'').length} chars)`,r.reasoning?`<pre>${esc(r.reasoning)}</pre>`:'')}
      </div></div></div>`}).join('');
}
$('list').addEventListener('toggle',e=>{const k=e.target.dataset&&e.target.dataset.k;if(k)manual.set(k,e.target.open)},true);
$('list').addEventListener('click',e=>{if(e.target.tagName==='IMG'){$('zoom').querySelector('img').src=e.target.src;$('zoom').style.display='flex'}});
$('zoom').onclick=()=>$('zoom').style.display='none';
document.querySelectorAll('[data-x]').forEach(b=>b.onclick=()=>{S.open[b.dataset.x]=!S.open[b.dataset.x];manual.clear();syncButtons();save();tick(true)});
$('all').onclick=()=>{for(const k in S.open)S.open[k]=true;manual.clear();syncButtons();save();tick(true)};
$('none').onclick=()=>{for(const k in S.open)S.open[k]=false;manual.clear();syncButtons();save();tick(true)};
$('img').onclick=()=>{S.img=!S.img;syncButtons();save();tick(true)};
['run','cell','label','sort'].forEach(i=>$(i).onchange=()=>{save();tick(true)});
$('cols').onchange=()=>{layout();save()};
syncButtons();layout();tick(true);setInterval(tick,2000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path.startswith("/data"):
            body, ctype = json.dumps(runs()).encode(), "application/json"
        elif path.startswith("/images/"):
            f = os.path.realpath(os.path.join(HERE, path.lstrip("/")))
            if not f.startswith(os.path.realpath(IMAGES) + os.sep) or not os.path.isfile(f):
                self.send_error(404)
                return
            with open(f, "rb") as fh:
                body = fh.read()
            ctype = mimetypes.guess_type(f)[0] or "application/octet-stream"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"monitor on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
