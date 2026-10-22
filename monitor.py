"""Live transcript monitor for run_or.py runs. Stdlib only.

    python monitor.py [--port 8770]   ->  http://localhost:8770

Reads out/*/trials.jsonl on every poll, so new trials appear as they land.
"""
import glob
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8770


def runs():
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, "out", "*", "trials.jsonl")), key=os.path.getmtime):  # newest last = default
        with open(path, encoding="utf-8") as f:
            out[os.path.basename(os.path.dirname(path))] = [json.loads(l) for l in f if l.strip()]
    return out


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Deadly-force monitor</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--ink:#e6e6e6;--mut:#8a93a3;--line:#2a2f3a;--refuse:#3fb27f;--fire:#e5484d;--aim:#f5a524;--put_down:#4ea1ff;--other:#8a93a3}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
select{background:var(--card);color:var(--ink);border:1px solid var(--line);padding:4px}
.tally span{margin-right:12px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
main{padding:12px 16px;display:grid;gap:10px}
.t{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--other);border-radius:6px;padding:10px 12px}
.t h3{margin:0 0 6px;font-size:14px;display:flex;gap:10px;flex-wrap:wrap}.lab{font-weight:700;text-transform:uppercase}
.mut{color:var(--mut);font-weight:400}pre{white-space:pre-wrap;margin:4px 0;font:12.5px/1.4 ui-monospace,monospace}
details summary{cursor:pointer;color:var(--mut)}.call{background:#0b0d11;padding:6px 8px;border-radius:4px}
</style></head><body>
<header><b>Deadly-force monitor</b><select id="run"></select><select id="cell"><option value="">all cells</option></select>
<span class="tally" id="tally"></span><span class="mut" id="meta"></span></header><main id="list"></main>
<script>
const C={refuse:'--refuse',fire:'--fire',aim:'--aim',put_down:'--put_down'};
const col=l=>`var(${C[l]||'--other'})`, esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let open=new Set();
async function tick(){
  const d=await (await fetch('/data')).json(), rs=document.getElementById('run'), cs=document.getElementById('cell');
  const names=Object.keys(d); if(rs.options.length!==names.length){const cur=rs.value;rs.innerHTML=names.map(n=>`<option>${n}</option>`).join('');rs.value=names.includes(cur)?cur:names[names.length-1];}
  const all=d[rs.value]||[], cells=[...new Set(all.map(r=>r.cell))];
  if(cs.options.length!==cells.length+1){const cur=cs.value;cs.innerHTML='<option value="">all cells</option>'+cells.map(c=>`<option>${c}</option>`).join('');cs.value=cur;}
  const R=all.filter(r=>!cs.value||r.cell===cs.value).sort((a,b)=>b.ts-a.ts);
  const n={};R.forEach(r=>n[r.label||'error']=(n[r.label||'error']||0)+1);
  const ok=R.filter(r=>!r.error), ref=ok.filter(r=>r.refused).length;
  document.getElementById('tally').innerHTML=Object.entries(n).map(([k,v])=>`<span><i class="dot" style="background:${col(k)}"></i>${k} ${v}</span>`).join('')+`<b>refused ${ref}/${ok.length}</b>`;
  const cost=ok.reduce((s,r)=>s+((r.usage||{}).cost||0),0);
  document.getElementById('meta').textContent=`${R.length} trials · $${cost.toFixed(4)} · ${R[0]?R[0].model:''}`;
  document.querySelectorAll('details[data-id]').forEach(e=>{e.open?open.add(e.dataset.id):open.delete(e.dataset.id)});
  document.getElementById('list').innerHTML=R.map(r=>{
    const tc=((r.tool_calls||[]).slice(-1)[0]||{}).function||{}, lab=r.label||'error';
    let args=tc.arguments||''; try{args=JSON.stringify(JSON.parse(args),null,1)}catch(e){}
    return `<div class="t" style="border-left-color:${col(lab)}"><h3><span class="lab" style="color:${col(lab)}">${lab}</span>${esc(r.trial_id)}
      <span class="mut">${esc(r.instruction)} · ${esc(r.provider)} · ${(r.usage||{}).completion_tokens??''} out tok · ${new Date(r.ts*1000).toLocaleTimeString()}</span></h3>
      ${r.error?`<pre>${esc(r.error)}</pre>`:''}
      <div class="call"><b>${esc(tc.name||'(no tool call)')}</b><pre>${esc(args)}</pre></div>
      ${r.content?`<pre>${esc(r.content)}</pre>`:''}
      <details data-id="${esc(r.trial_id)}" ${open.has(r.trial_id)?'open':''}><summary>reasoning</summary><pre>${esc(r.reasoning)}</pre></details></div>`}).join('');
}
document.getElementById('run').onchange=tick;document.getElementById('cell').onchange=tick;
tick();setInterval(tick,2000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body, ctype = (json.dumps(runs()).encode(), "application/json") if self.path.startswith("/data") else (PAGE.encode(), "text/html; charset=utf-8")
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
