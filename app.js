'use strict';

const DB_NAME = 'shoe-match-db';
const DB_VERSION = 5;
const SAMPLE_STORE = 'shoes';
const BATCH_STORE = 'importBatches';
const FEATURE_VERSION = 'dinov2-small-cls-local-color-v2';
const MODEL_URL = './models/dinov2-small/model_quantized.onnx';
const MODEL_DIMENSION = 384;
const MODEL_INPUT_SIZE = 224;
const MODEL_RESIZE_SHORT = 256;
const LOCAL_GRID = 4;
const LOCAL_REGIONS = LOCAL_GRID * LOCAL_GRID;
const LOCAL_FEATURE_LENGTH = LOCAL_REGIONS * MODEL_DIMENSION;
const COLOR_FEATURE_LENGTH = 89;
let db;
let stockFile = null;
let excelFile = null;
let queryToken = 0;
let modelSessionPromise = null;

const $ = s => document.querySelector(s);
const sleepFrame = () => new Promise(r => requestAnimationFrame(r));
const wait = ms => new Promise(r => setTimeout(r, ms));
const reqP = req => new Promise((res, rej) => { req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error); });

async function openDB(){
  return new Promise((resolve,reject)=>{
    const req=indexedDB.open(DB_NAME,DB_VERSION);
    req.onupgradeneeded=()=>{
      const d=req.result;
      let samples;
      if(!d.objectStoreNames.contains(SAMPLE_STORE)){
        samples=d.createObjectStore(SAMPLE_STORE,{keyPath:'id',autoIncrement:true});
      } else samples=req.transaction.objectStore(SAMPLE_STORE);
      if(!samples.indexNames.contains('code')) samples.createIndex('code','code',{unique:false});
      if(!samples.indexNames.contains('createdAt')) samples.createIndex('createdAt','createdAt',{unique:false});
      if(!samples.indexNames.contains('batchId')) samples.createIndex('batchId','batchId',{unique:false});

      let batches;
      if(!d.objectStoreNames.contains(BATCH_STORE)){
        batches=d.createObjectStore(BATCH_STORE,{keyPath:'batchId'});
      } else batches=req.transaction.objectStore(BATCH_STORE);
      if(!batches.indexNames.contains('startedAt')) batches.createIndex('startedAt','startedAt',{unique:false});
      if(!batches.indexNames.contains('fingerprint')) batches.createIndex('fingerprint','fingerprint',{unique:false});
    };
    req.onsuccess=()=>resolve(req.result);
    req.onerror=()=>reject(req.error);
  });
}
function store(name,mode='readonly'){return db.transaction(name,mode).objectStore(name)}
async function getAllSamples(){return reqP(store(SAMPLE_STORE).getAll())}
async function addSample(v){return reqP(store(SAMPLE_STORE,'readwrite').add(v))}
async function deleteSample(id){return reqP(store(SAMPLE_STORE,'readwrite').delete(id))}
async function countAll(){return reqP(store(SAMPLE_STORE).count())}
async function getAllBatches(){return reqP(store(BATCH_STORE).getAll())}
async function putBatch(v){return reqP(store(BATCH_STORE,'readwrite').put(v))}
async function clearSamples(){return reqP(store(SAMPLE_STORE,'readwrite').clear())}

function batchId(){return `${Date.now()}-${crypto.randomUUID?.()||Math.random().toString(36).slice(2)}`}
function fileFingerprint(file){return `${file.name}|${file.size}|${file.lastModified||0}`}
function formatTime(ts){if(!ts)return '-';return new Date(ts).toLocaleString('zh-CN',{hour12:false})}
function formatBytes(bytes){if(!Number.isFinite(bytes))return '-';if(bytes<1024)return `${bytes} B`;if(bytes<1048576)return `${(bytes/1024).toFixed(1)} KB`;return `${(bytes/1048576).toFixed(1)} MB`}

function loadImage(fileOrUrl){
  return new Promise((resolve,reject)=>{
    const img=new Image();
    const objectUrl=typeof fileOrUrl==='string'?null:URL.createObjectURL(fileOrUrl);
    img.onload=()=>{if(objectUrl)URL.revokeObjectURL(objectUrl);resolve(img)};
    img.onerror=e=>{if(objectUrl)URL.revokeObjectURL(objectUrl);reject(e)};
    img.src=objectUrl||fileOrUrl;
  });
}
function drawCover(img,size=256){
  const c=document.createElement('canvas'); c.width=c.height=size;
  const ctx=c.getContext('2d',{willReadFrequently:true});
  const scale=Math.max(size/img.width,size/img.height),w=img.width*scale,h=img.height*scale;
  ctx.drawImage(img,(size-w)/2,(size-h)/2,w,h); return c;
}
function makeThumb(img){return drawCover(img,420).toDataURL('image/jpeg',0.76)}
function setModelStatus(text,state='idle'){
  const el=$('#modelStatus');
  if(!el)return;
  el.textContent=text;
  el.dataset.state=state;
}
async function ensureModel(statusEl){
  if(modelSessionPromise)return modelSessionPromise;
  if(!window.ort)throw Error('AI 运行库未加载，请刷新页面重试');
  setModelStatus('AI 模型加载中…','loading');
  if(statusEl)setStatus(statusEl,'首次使用正在加载本地 AI 模型（约 24 MB），后续会自动缓存…');
  const base=new URL('./vendor/onnxruntime/',location.href).href;
  ort.env.wasm.wasmPaths=base;
  ort.env.wasm.numThreads=1;
  // GitHub Pages does not provide cross-origin isolation. Keep the portable
  // single-thread WASM backend; yielding between samples keeps import progress visible.
  ort.env.wasm.proxy=false;
  modelSessionPromise=ort.InferenceSession.create(MODEL_URL,{executionProviders:['wasm'],graphOptimizationLevel:'all'})
    .then(session=>{setModelStatus('AI 模型已就绪','ready');return session})
    .catch(error=>{modelSessionPromise=null;setModelStatus('AI 模型加载失败','error');throw error});
  return modelSessionPromise;
}
function modelCanvas(img,mode='center',turns=0){
  const canvas=document.createElement('canvas');canvas.width=canvas.height=MODEL_INPUT_SIZE;
  const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.fillStyle='#fff';ctx.fillRect(0,0,MODEL_INPUT_SIZE,MODEL_INPUT_SIZE);
  const rotated=turns%2!==0,rw=rotated?img.height:img.width,rh=rotated?img.width:img.height;
  const scale=mode==='letterbox'?MODEL_INPUT_SIZE/Math.max(rw,rh):MODEL_RESIZE_SHORT/Math.min(rw,rh);
  ctx.translate(MODEL_INPUT_SIZE/2,MODEL_INPUT_SIZE/2);ctx.rotate(turns*Math.PI/2);
  ctx.drawImage(img,-img.width*scale/2,-img.height*scale/2,img.width*scale,img.height*scale);
  return canvas;
}
function inputFromCanvases(canvases){
  const plane=MODEL_INPUT_SIZE*MODEL_INPUT_SIZE,input=new Float32Array(canvases.length*plane*3),means=[.485,.456,.406],stds=[.229,.224,.225];
  canvases.forEach((canvas,b)=>{const rgba=canvas.getContext('2d',{willReadFrequently:true}).getImageData(0,0,MODEL_INPUT_SIZE,MODEL_INPUT_SIZE).data,base=b*plane*3;for(let p=0,i=0;i<plane;i++,p+=4){input[base+i]=(rgba[p]/255-means[0])/stds[0];input[base+plane+i]=(rgba[p+1]/255-means[1])/stds[1];input[base+plane*2+i]=(rgba[p+2]/255-means[2])/stds[2]}});
  return input;
}
function normalizeVector(values){let norm=0;for(const v of values)norm+=v*v;norm=Math.sqrt(norm)||1;for(let i=0;i<values.length;i++)values[i]/=norm;return values}
function pooledLocalFeatures(data,batchIndex,stride,quantize=false){
  const out=quantize?new Int8Array(LOCAL_FEATURE_LENGTH):new Float32Array(LOCAL_FEATURE_LENGTH),batchBase=batchIndex*stride;
  for(let gy=0;gy<LOCAL_GRID;gy++)for(let gx=0;gx<LOCAL_GRID;gx++){
    const region=gy*LOCAL_GRID+gx,vec=new Float32Array(MODEL_DIMENSION);
    for(let py=gy*4;py<gy*4+4;py++)for(let px=gx*4;px<gx*4+4;px++){const tokenBase=batchBase+(1+py*16+px)*MODEL_DIMENSION;for(let d=0;d<MODEL_DIMENSION;d++)vec[d]+=Number(data[tokenBase+d])}
    normalizeVector(vec);const base=region*MODEL_DIMENSION;for(let d=0;d<MODEL_DIMENSION;d++)out[base+d]=quantize?Math.max(-127,Math.min(127,Math.round(vec[d]*127))):vec[d];
  }
  return out;
}
async function modelFeatures(canvases,statusEl,{quantizeLocal=false}={}){
  const session=await ensureModel(statusEl),input=inputFromCanvases(canvases),batch=canvases.length;
  const outputs=await session.run({pixel_values:new ort.Tensor('float32',input,[batch,3,MODEL_INPUT_SIZE,MODEL_INPUT_SIZE])}),tensor=outputs.last_hidden_state||Object.values(outputs)[0];
  if(!tensor||tensor.data.length<batch*MODEL_DIMENSION)throw Error('AI 模型输出格式不正确');
  const stride=tensor.data.length/batch,embeddings=[],locals=[];
  for(let b=0;b<batch;b++){const embedding=new Float32Array(MODEL_DIMENSION);for(let i=0;i<MODEL_DIMENSION;i++)embedding[i]=Number(tensor.data[b*stride+i]);normalizeVector(embedding);embeddings.push(Array.from(embedding));locals.push(pooledLocalFeatures(tensor.data,b,stride,quantizeLocal))}
  return {embeddings,locals};
}
function percentile(values,p){if(!values.length)return 0;values.sort((a,b)=>a-b);return values[Math.min(values.length-1,Math.max(0,Math.round((values.length-1)*p)))]}
function median(values){return percentile(values,.5)}
function rgbLabHsv(r8,g8,b8){
  const r0=r8/255,g0=g8/255,b0=b8/255,lin=x=>x<=.04045?x/12.92:Math.pow((x+.055)/1.055,2.4),r=lin(r0),g=lin(g0),b=lin(b0);
  const x=(.4124564*r+.3575761*g+.1804375*b)/.95047,y=.2126729*r+.7151522*g+.072175*b,z=(.0193339*r+.119192*g+.9503041*b)/1.08883,f=t=>t>.008856?Math.cbrt(t):7.787*t+16/116;
  const fy=f(y),lab=[(116*fy-16)*2.55,500*(f(x)-fy)+128,200*(fy-f(z))+128],max=Math.max(r0,g0,b0),min=Math.min(r0,g0,b0),delta=max-min;
  let hue=0;if(delta){if(max===r0)hue=60*((g0-b0)/delta%6);else if(max===g0)hue=60*((b0-r0)/delta+2);else hue=60*((r0-g0)/delta+4);if(hue<0)hue+=360}
  return {lab,h:hue/2,s:max?delta/max:0,v:max};
}
function morph(mask,w,h,dilate){const out=new Uint8Array(mask.length);for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){let value=dilate?0:1;for(let yy=y-1;yy<=y+1;yy++)for(let xx=x-1;xx<=x+1;xx++)value=dilate?(value||mask[yy*w+xx]):(value&&mask[yy*w+xx]);out[y*w+x]=value?1:0}return out}
function colorDescriptor(img){
  const scale=Math.min(1,320/Math.max(img.width,img.height)),w=Math.max(1,Math.round(img.width*scale)),h=Math.max(1,Math.round(img.height*scale)),canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(img,0,0,w,h);
  const rgba=ctx.getImageData(0,0,w,h).data,n=w*h,labs=new Float32Array(n*3),hs=new Float32Array(n),ss=new Float32Array(n),vs=new Float32Array(n),bw=Math.max(2,Math.round(Math.min(w,h)*.04)),border=[];
  for(let i=0,p=0;i<n;i++,p+=4){const c=rgbLabHsv(rgba[p],rgba[p+1],rgba[p+2]);labs[i*3]=c.lab[0];labs[i*3+1]=c.lab[1];labs[i*3+2]=c.lab[2];hs[i]=c.h;ss[i]=c.s;vs[i]=c.v;const x=i%w,y=Math.floor(i/w);if(x<bw||x>=w-bw||y<bw||y>=h-bw)border.push(i)}
  const bg=[0,1,2].map(ch=>median(border.map(i=>labs[i*3+ch]))),bgSat=median(border.map(i=>ss[i])),dist=new Float32Array(n),borderDist=[];
  for(let i=0;i<n;i++){const a=labs[i*3]-bg[0],b=labs[i*3+1]-bg[1],c=labs[i*3+2]-bg[2];dist[i]=Math.sqrt(a*a+b*b+c*c);if((i%w)<bw||(i%w)>=w-bw||i<w*bw||i>=w*(h-bw))borderDist.push(dist[i])}
  const threshold=Math.max(18,percentile(borderDist,.92)+8);let mask=new Uint8Array(n);for(let i=0;i<n;i++)mask[i]=(dist[i]>threshold||(ss[i]>bgSat+.10&&dist[i]>10)||vs[i]<bg[0]/255-.16)?1:0;
  mask=morph(morph(mask,w,h,false),w,h,true);for(let k=0;k<3;k++)mask=morph(mask,w,h,true);for(let k=0;k<3;k++)mask=morph(mask,w,h,false);
  const labels=new Int32Array(n),queue=new Int32Array(n),components=[];let label=0;
  for(let start=0;start<n;start++){if(!mask[start]||labels[start])continue;label++;let head=0,tail=0,area=0,sx=0,sy=0;queue[tail++]=start;labels[start]=label;while(head<tail){const i=queue[head++],x=i%w,y=Math.floor(i/w);area++;sx+=x;sy+=y;for(const j of [i-1,i+1,i-w,i+w])if(j>=0&&j<n&&mask[j]&&!labels[j]&&Math.abs((j%w)-x)<=1){labels[j]=label;queue[tail++]=j}}const cx=sx/area,cy=sy/area,central=Math.max(0,1-1.3*Math.hypot((cx-w/2)/w,(cy-h/2)/h));components.push({label,area,score:area*(.35+central)})}
  components.sort((a,b)=>b.score-a.score);const keep=new Set(components.slice(0,2).filter(x=>x.area>.002*n).map(x=>x.label)),selected=[];for(let i=0;i<n;i++)if(keep.has(labels[i])&&dist[i]>Math.max(8,threshold*.45))selected.push(i);
  if(selected.length<100){const fallback=[...dist].sort((a,b)=>a-b)[Math.floor(n*.75)];selected.length=0;for(let i=0;i<n;i++)if(dist[i]>fallback)selected.push(i)}
  const parts=[],hist=(bins,min,max,value,weight)=>{const out=new Float32Array(bins);for(const i of selected){const v=value(i),slot=Math.max(0,Math.min(bins-1,Math.floor((v-min)/(max-min)*bins)));out[slot]+=weight?weight(i):1}let sum=0;for(const v of out)sum+=v;for(const v of out)parts.push(Math.sqrt(v/(sum||1)))};
  hist(18,0,180,i=>hs[i],i=>ss[i]*ss[i]+.05);hist(10,0,1,i=>ss[i]);hist(12,0,1,i=>vs[i]);hist(12,70,190,i=>labs[i*3+1]);hist(12,70,210,i=>labs[i*3+2]);
  for(const getter of [i=>labs[i*3],i=>labs[i*3+1],i=>labs[i*3+2],i=>ss[i]*255,i=>vs[i]*255]){const values=selected.map(getter);for(const p of [.1,.25,.5,.75,.9])parts.push(percentile(values,p)/255)}
  normalizeVector(parts);return {vector:parts,foregroundValue:median(selected.map(i=>vs[i]))};
}
function embeddingSimilarity(a,b){
  if(!a||!b||a.length!==MODEL_DIMENSION||b.length!==MODEL_DIMENSION)return -1;
  let dot=0;for(let i=0;i<MODEL_DIMENSION;i++)dot+=Number(a[i])*Number(b[i]);
  return Math.max(0,Math.min(1,dot));
}
function colorSimilarity(a,b){if(!a||!b||a.length!==COLOR_FEATURE_LENGTH||b.length!==COLOR_FEATURE_LENGTH)return 0;let dot=0;for(let i=0;i<a.length;i++)dot+=Number(a[i])*Number(b[i]);return Math.max(0,Math.min(1,dot))}
function localSimilarity(queryLocals,gallery){
  if(!gallery||gallery.length!==LOCAL_FEATURE_LENGTH)return 0;let best=-1;
  for(const query of queryLocals){const rowBest=new Float32Array(LOCAL_REGIONS).fill(-1),colBest=new Float32Array(LOCAL_REGIONS).fill(-1);
    for(let qr=0;qr<LOCAL_REGIONS;qr++)for(let gr=0;gr<LOCAL_REGIONS;gr++){let dot=0,qb=qr*MODEL_DIMENSION,gb=gr*MODEL_DIMENSION;for(let d=0;d<MODEL_DIMENSION;d++)dot+=query[qb+d]*gallery[gb+d]/127;if(dot>rowBest[qr])rowBest[qr]=dot;if(dot>colBest[gr])colBest[gr]=dot}
    const topMean=values=>[...values].sort((a,b)=>b-a).slice(0,6).reduce((a,b)=>a+b,0)/6,score=(topMean(rowBest)+topMean(colBest))/2;if(score>best)best=score;
  }
  return Math.max(0,Math.min(1,best));
}
function familyKey(code){const text=String(code||'').trim().toLocaleLowerCase(),match=text.match(/^(\d+)/);return match?match[1]:text}
function isWhiteVariant(code){return /(^|[\s._/-])white([\s._/-]|$)/i.test(String(code||''))}
async function queryFeatures(img,statusEl){
  const canvases=[];for(let turns=0;turns<4;turns++){canvases.push(modelCanvas(img,'center',turns),modelCanvas(img,'letterbox',turns))}
  const model=await modelFeatures(canvases,statusEl),color=colorDescriptor(img);return {embeddings:model.embeddings,locals:model.locals,colorFeature:color.vector,foregroundValue:color.foregroundValue};
}
async function featuresFromFile(file,statusEl){const img=await loadImage(file),model=await modelFeatures([modelCanvas(img)],statusEl,{quantizeLocal:true}),color=colorDescriptor(img);return {img,embedding:model.embeddings[0],localFeatures:model.locals[0],colorFeature:color.vector,featureVersion:FEATURE_VERSION,thumb:makeThumb(img)}}
function showPreview(file,el){el.src=URL.createObjectURL(file);el.classList.remove('hidden')}
async function refreshCount(){
  const total=await countAll();$('#dbCount').textContent=`${total} 条`;
  const all=total?await getAllSamples():[],searchable=all.filter(x=>x.featureVersion===FEATURE_VERSION&&x.embedding?.length===MODEL_DIMENSION&&x.localFeatures?.length===LOCAL_FEATURE_LENGTH&&x.colorFeature?.length===COLOR_FEATURE_LENGTH).length;
  if(searchable)setModelStatus(`AI 特征 ${searchable} 条`,'ready');
  else if(total)setModelStatus('旧版数据需重新导入','warning');
  else setModelStatus('AI 模型按需加载','idle');
  await refreshStorage();
}
async function refreshStorage(){
  const el=$('#storageUsage');if(!el)return;
  try{const est=await navigator.storage?.estimate?.();el.textContent=est?`浏览器存储：已用 ${formatBytes(est.usage)} / 配额 ${formatBytes(est.quota)}`:'浏览器未提供存储估算';}
  catch{el.textContent='浏览器未提供存储估算'}
}
function setStatus(el,text,error=false){el.textContent=text;el.style.color=error?'#b91c1c':'#475569'}
function esc(v){return String(v??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
function normalizedRecord(x){return {...x,code:String(x.code??x.shoeNo??'').trim(),customerNo:x.customerNo??'',orderNo:x.orderNo??'',sendDate:x.sendDate??'',remark:x.remark??'',createdAt:x.createdAt||Date.now()}}

function renderResults(items){
  const root=$('#results');root.innerHTML='';
  if(!items.length){root.innerHTML='<div class="empty">数据库暂无样品，请先导入 Excel 或单条入库。</div>';return}
  items.forEach((x,i)=>{
    const n=$('#resultTemplate').content.cloneNode(true);n.querySelector('.rank').textContent=i+1;const resultImg=n.querySelector('.result-img');
    resultImg.src=x.thumb;
    resultImg.onclick=()=>openImageModal(x.thumb);n.querySelector('.result-code').textContent=`样品编号：${x.code||'未填写'}`;
    const meta=[];if(x.sendDate)meta.push(['寄出时间',x.sendDate]);if(x.customerNo)meta.push(['客户编号',x.customerNo]);if(x.orderNo)meta.push(['订单编号',x.orderNo]);if(x.remark)meta.push(['特别要求',x.remark]);
    n.querySelector('.meta-list').innerHTML=meta.length?meta.map(([k,v])=>`<p><b>${esc(k)}：</b>${esc(v)}</p>`).join(''):'<p class="muted">无其他资料</p>';
    const pct=Math.round(x.score*1000)/10;n.querySelector('.result-score').textContent=`${pct} / 100`;n.querySelector('.score-bar i').style.width=`${pct}%`;root.appendChild(n);
  });
}
async function handleQuery(file){
  const token=++queryToken;
  const status=$('#queryStatus');
  const results=$('#results');
  results.innerHTML='';
  setStatus(status,'正在处理当前查询图片…');
  const start=performance.now();
  try{
    const img=await loadImage(file),query=await queryFeatures(img,status);
    if(token!==queryToken)return;
    const stored=await getAllSamples(),all=stored.filter(x=>x.featureVersion===FEATURE_VERSION&&x.embedding?.length===MODEL_DIMENSION&&x.localFeatures?.length===LOCAL_FEATURE_LENGTH&&x.colorFeature?.length===COLOR_FEATURE_LENGTH);
    if(!all.length){
      if(stored.length)throw Error('现有数据库是旧版特征，请在“批量与管理”中清空后重新导入 Excel');
      throw Error('数据库暂无样品，请先导入 Excel');
    }
    setStatus(status,`正在进行鞋型、局部细节与颜色比对（${all.length} 条）…`);
    await sleepFrame();
    const samples=[],families=new Map();
    for(let i=0;i<all.length;i++){
      if(token!==queryToken)return;
      const s=normalizedRecord(all[i]),global=Math.max(...query.embeddings.map(e=>embeddingSimilarity(e,s.embedding))),color=colorSimilarity(query.colorFeature,s.colorFeature),family=familyKey(s.code)||`__${s.id}`,item={record:s,global,color,family,local:0};samples.push(item);
      const current=families.get(family)||{key:family,global:0,local:0,items:[]};current.global=Math.max(current.global,global);current.items.push(item);families.set(family,current);
      if(i&&i%300===0)await sleepFrame();
    }
    const shortlist=[...families.values()].sort((a,b)=>b.global-a.global).slice(0,80),shortKeys=new Set(shortlist.map(x=>x.key));
    setStatus(status,`已锁定 ${shortlist.length} 个鞋型，正在核对鞋底、拉链和饰条细节…`);await sleepFrame();
    let checked=0;for(const item of samples)if(shortKeys.has(item.family)){item.local=localSimilarity(query.locals,item.record.localFeatures);const family=families.get(item.family);family.local=Math.max(family.local,item.local);if(++checked%20===0)await sleepFrame()}
    const scored=[];for(const family of shortlist){family.score=.2*family.global+.8*(family.local||family.global);const variants=new Map();for(const item of family.items){const key=item.record.code.trim().toLocaleLowerCase()||`__${item.record.id}`,v=variants.get(key)||{code:key,global:0,color:0,items:[]};v.global=Math.max(v.global,item.global);v.color=Math.max(v.color,item.color);v.items.push(item);variants.set(key,v)}
      let winner=null;for(const variant of variants.values()){variant.score=.7*variant.global+.3*variant.color;if(query.foregroundValue<.62&&isWhiteVariant(variant.code))variant.score-=.06;if(!winner||variant.score>winner.score)winner=variant}
      const representative=winner.items.sort((a,b)=>(.7*b.global+.3*b.color)-(.7*a.global+.3*a.color))[0].record;scored.push({...representative,score:family.score,variantScore:winner.score});
    }
    scored.sort((a,b)=>b.score-a.score);
    if(token!==queryToken)return;
    renderResults(scored.slice(0,5));
    const top=scored[0]?.code?`第一名：${scored[0].code}。`:'';
    setStatus(status,`完成：比对 ${all.length} 条、合并为 ${families.size} 个鞋型，耗时 ${Math.round(performance.now()-start)} ms。${top}`);
  }catch(e){
    console.error(e);
    setStatus(status,e.message||'识别失败，请换一张图片重试。',true);
  }
}
function openImageModal(src){
  const m=$('#imageModal'),img=$('#modalImage');
  if(!m||!img)return;
  img.src=src;
  m.classList.remove('hidden');
}
function closeImageModal(){
  const m=$('#imageModal');
  if(m)m.classList.add('hidden');
}

async function renderInventory(){
  const root=$('#inventoryList');root.innerHTML='';const all=(await getAllSamples()).map(normalizedRecord).sort((a,b)=>b.createdAt-a.createdAt).slice(0,100);
  if(!all.length){root.innerHTML='<div class="empty">暂无样品数据</div>';return}
  all.forEach(x=>{const d=document.createElement('article');d.className='inventory-item';d.innerHTML='<img alt="样品照"><div><h4></h4><p></p></div><button class="mini-danger">删除</button>';d.querySelector('img').src=x.thumb;d.querySelector('h4').textContent=x.code||'未填写编号';d.querySelector('p').textContent=[x.customerNo&&`客户：${x.customerNo}`,x.orderNo&&`订单：${x.orderNo}`,x.sendDate].filter(Boolean).join(' · ')||'单条入库';d.querySelector('button').onclick=async()=>{if(confirm(`确定删除编号 ${x.code||'未填写'} 的这一条记录？`)){await deleteSample(x.id);await refreshCount();renderInventory();renderBatches()}};root.appendChild(d)});
}
function downloadJSON(name,data){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}

function xml(text){return new DOMParser().parseFromString(text,'application/xml')}
function localChildren(node,name){return [...node.getElementsByTagNameNS('*',name)]}
function pathJoin(base,target){const rooted=String(target).startsWith('/'),parts=((rooted?'':base)+'/'+target).split('/'),out=[];for(const p of parts){if(!p||p==='.')continue;if(p==='..')out.pop();else out.push(p)}return out.join('/')}
function colToIndex(ref){let n=0;for(const ch of ref.match(/[A-Z]+/i)[0].toUpperCase())n=n*26+ch.charCodeAt(0)-64;return n-1}
function indexToCol(index){let n=index+1,out='';while(n>0){n--;out=String.fromCharCode(65+n%26)+out;n=Math.floor(n/26)}return out}
function excelDate(v){if(v===''||v==null)return '';const n=Number(v);if(!Number.isFinite(n))return String(v);const date=new Date(Date.UTC(1899,11,30)+n*86400000);return date.toISOString().slice(0,10)}
async function parseExcelPackage(file){
  if(!window.JSZip)throw Error('Excel 解压组件未加载');
  const zip=await JSZip.loadAsync(file);
  const ssFile=zip.file('xl/sharedStrings.xml');let shared=[];
  if(ssFile){const doc=xml(await ssFile.async('text'));shared=localChildren(doc,'si').map(si=>localChildren(si,'t').map(t=>t.textContent).join(''))}
  const wb=xml(await zip.file('xl/workbook.xml').async('text')),wbRels=xml(await zip.file('xl/_rels/workbook.xml.rels').async('text'));
  const firstSheet=localChildren(wb,'sheet')[0],sheetRid=firstSheet.getAttributeNS('http://schemas.openxmlformats.org/officeDocument/2006/relationships','id')||firstSheet.getAttribute('r:id');
  const wbRel=localChildren(wbRels,'Relationship').find(r=>r.getAttribute('Id')===sheetRid);const sheetPath=pathJoin('xl',wbRel.getAttribute('Target'));
  const sheetDoc=xml(await zip.file(sheetPath).async('text'));const rows=[];
  for(const row of localChildren(sheetDoc,'row')){const obj={_row:Number(row.getAttribute('r'))};for(const c of [...row.children].filter(n=>n.localName==='c')){const ref=c.getAttribute('r'),idx=colToIndex(ref),type=c.getAttribute('t'),v=localChildren(c,'v')[0]?.textContent??'',inline=localChildren(c,'t').map(t=>t.textContent).join('');obj[idx]=type==='s'?shared[Number(v)]??'':type==='inlineStr'?inline:v}rows.push(obj)}
  const header=rows.shift()||{};const names={};Object.keys(header).filter(k=>!k.startsWith('_')).forEach(k=>names[String(header[k]).trim()]=Number(k));
  const sheetRelsPath=sheetPath.replace(/([^/]+)$/,'_rels/$1.rels');let drawingPath='xl/drawings/drawing1.xml';
  const sheetRelsFile=zip.file(sheetRelsPath);if(sheetRelsFile){const rd=xml(await sheetRelsFile.async('text'));const rel=localChildren(rd,'Relationship').find(r=>(r.getAttribute('Type')||'').endsWith('/drawing'));if(rel)drawingPath=pathJoin(sheetPath.split('/').slice(0,-1).join('/'),rel.getAttribute('Target'))}
  const imageColumns=new Set(Object.entries(names).filter(([name])=>/^(图片|照片|图像)\s*\d*$/i.test(name)).map(([,index])=>index));
  const imagesByRow=new Map();const drawingFile=zip.file(drawingPath);
  if(drawingFile){const drawingDoc=xml(await drawingFile.async('text')),relsPath=drawingPath.replace(/([^/]+)$/,'_rels/$1.rels'),relsFile=zip.file(relsPath);if(relsFile){const relsDoc=xml(await relsFile.async('text')),relMap=new Map(localChildren(relsDoc,'Relationship').map(r=>[r.getAttribute('Id'),pathJoin(drawingPath.split('/').slice(0,-1).join('/'),r.getAttribute('Target'))]));
      for(const anchor of [...drawingDoc.documentElement.children]){const from=[...anchor.children].find(n=>n.localName==='from'),pic=localChildren(anchor,'pic')[0];if(!from||!pic)continue;const row0=Number(localChildren(from,'row')[0]?.textContent),col0=Number(localChildren(from,'col')[0]?.textContent),blip=localChildren(pic,'blip')[0],rid=blip?.getAttributeNS('http://schemas.openxmlformats.org/officeDocument/2006/relationships','embed')||blip?.getAttribute('r:embed'),mediaPath=relMap.get(rid);if(!mediaPath||(imageColumns.size&&!imageColumns.has(col0)))continue;const rowImages=imagesByRow.get(row0+1)||[];rowImages.push({col:col0,mediaPath});imagesByRow.set(row0+1,rowImages)}
      for(const rowImages of imagesByRow.values())rowImages.sort((a,b)=>a.col-b.col);
    }}
  return {zip,rows,names,imagesByRow};
}
function pick(obj,names,candidates){for(const name of candidates){const idx=names[name];if(idx!==undefined)return obj[idx]??''}return ''}

async function findDuplicateBatch(file){
  const fp=fileFingerprint(file);return (await getAllBatches()).filter(x=>x.fingerprint===fp&&x.status==='completed'&&(x.successCount||0)>0).sort((a,b)=>b.startedAt-a.startedAt)[0];
}
async function deleteBatchData(batchIdToDelete){
  return new Promise((resolve,reject)=>{
    const t=db.transaction([SAMPLE_STORE,BATCH_STORE],'readwrite');
    const samples=t.objectStore(SAMPLE_STORE),idx=samples.index('batchId');
    const cursor=idx.openCursor(IDBKeyRange.only(batchIdToDelete));let deleted=0;
    cursor.onsuccess=()=>{const c=cursor.result;if(c){c.delete();deleted++;c.continue()}};
    cursor.onerror=()=>reject(cursor.error);
    t.oncomplete=()=>resolve(deleted);t.onerror=()=>reject(t.error);
  });
}
async function markBatchDeleted(batch,deletedCount){
  await putBatch({...batch,status:'deleted',deletedAt:Date.now(),deletedCount,activeCount:0});
}
async function removeBatch(batch){
  if(!confirm(`确定删除批次“${batch.fileName}”中的 ${batch.successCount||0} 条样品？\n\n该批次的导入审计记录会保留，样品数据将删除。`))return;
  const deleted=await deleteBatchData(batch.batchId);await markBatchDeleted(batch,deleted);await refreshCount();await renderBatches();await renderInventory();setStatus($('#manageStatus'),`已删除批次数据 ${deleted} 条，审计记录已保留。`);
}
async function renderBatches(){
  const root=$('#batchList');if(!root)return;root.innerHTML='';
  const batches=(await getAllBatches()).sort((a,b)=>b.startedAt-a.startedAt);
  if(!batches.length){root.innerHTML='<div class="empty compact">暂无 Excel 导入记录</div>';return}
  for(const b of batches){
    let activeCount=0;if(b.status!=='deleted')activeCount=await reqP(store(SAMPLE_STORE).index('batchId').count(IDBKeyRange.only(b.batchId)));
    const card=document.createElement('article');card.className=`batch-card ${b.status==='deleted'?'batch-deleted':''}`;
    const statusText=b.status==='deleted'?'已删除':b.status==='completed'?'导入完成':b.status==='failed'?'导入失败':'处理中断';
    card.innerHTML=`<div class="batch-main"><div class="batch-title"><strong>${esc(b.fileName||'未知文件')}</strong><span class="batch-status">${statusText}</span></div><p>${formatTime(b.startedAt)} · 文件 ${formatBytes(b.fileSize)}</p><div class="batch-stats"><span>总计 ${b.totalCount||0}</span><span>成功 ${b.successCount||0}</span><span>失败 ${b.failedCount||0}</span><span>当前保留 ${activeCount}</span></div></div><div class="batch-actions"></div>`;
    const actions=card.querySelector('.batch-actions');
    if((b.failures||[]).length){const report=document.createElement('button');report.className='mini-secondary';report.textContent='下载报告';report.onclick=()=>downloadJSON(`导入报告-${b.fileName}-${b.batchId}.json`,b);actions.appendChild(report)}
    if(b.status!=='deleted'){const del=document.createElement('button');del.className='mini-danger';del.textContent='删除本批';del.onclick=()=>removeBatch(b);actions.appendChild(del)}
    root.appendChild(card);
  }
}

async function importExcel(file){
  const duplicate=await findDuplicateBatch(file);
  if(duplicate){
    const again=confirm(`检测到相同文件可能已经导入：\n${duplicate.fileName}\n${formatTime(duplicate.startedAt)}\n成功 ${duplicate.successCount||0} 条。\n\n点击“确定”仍作为新批次导入；点击“取消”停止。`);
    if(!again)return;
  }
  const id=batchId(),startedAt=Date.now(),fingerprint=fileFingerprint(file);
  const btn=$('#excelImportBtn'),status=$('#excelStatus'),wrap=$('#excelProgressWrap'),bar=$('#excelProgressBar'),txt=$('#excelProgressText');btn.disabled=true;wrap.classList.remove('hidden');bar.style.width='0%';setStatus(status,'正在准备 AI 模型…');
  let success=0,failed=0,total=0;const failures=[];
  let batch={batchId:id,fileName:file.name,fileSize:file.size,fingerprint,startedAt,status:'processing',totalCount:0,successCount:0,failedCount:0,failures:[]};await putBatch(batch);
  try{
    await ensureModel(status);setStatus(status,'正在解析 Excel 并计算 AI 鞋型、局部与颜色特征，请勿关闭页面…');
    const {zip,rows,names,imagesByRow}=await parseExcelPackage(file),candidates=[];
    for(const r of rows){const code=String(pick(r,names,['样品编号','鞋子编号','编号'])).trim(),rowImages=imagesByRow.get(r._row)||[];if(!code&&!rowImages.length)continue;if(rowImages.length){for(const image of rowImages)candidates.push({row:r,code,...image})}else candidates.push({row:r,code,col:null,mediaPath:null})}
    total=candidates.length;batch.totalCount=total;await putBatch(batch);
    for(let i=0;i<candidates.length;i++){
      const {row:r,code,col,mediaPath}=candidates[i],imageColumn=col==null?'':indexToCol(col);
      try{
        if(!code)throw Error('样品编号为空');if(!mediaPath)throw Error('缺少内嵌图片');const zf=zip.file(mediaPath);if(!zf)throw Error('找不到图片文件');const blob=await zf.async('blob');const {embedding,localFeatures,colorFeature,featureVersion,thumb}=await featuresFromFile(blob);
        await addSample({code,sendDate:excelDate(pick(r,names,['样品寄出时间','寄出时间','日期'])),customerNo:String(pick(r,names,['客户编号','客户号'])).trim(),orderNo:String(pick(r,names,['订单编号','订单号'])).trim(),remark:String(pick(r,names,['特别要求','备注','要求'])).trim(),thumb,embedding,localFeatures,colorFeature,featureVersion,source:'excel',sourceFile:file.name,sourceRow:r._row,sourceImageColumn:imageColumn,batchId:id,createdAt:Date.now()});success++;
      }catch(e){failed++;failures.push({row:r._row,column:imageColumn,code,error:e.message||String(e)});console.warn('导入图片失败',r._row,imageColumn,e)}
      const pct=Math.round((i+1)/candidates.length*100);bar.style.width=`${pct}%`;txt.textContent=`${i+1} / ${candidates.length} · AI 特征成功 ${success} · 失败 ${failed}`;
      if(i%3===0){await sleepFrame();await wait(5)}
      if(i%25===0){batch={...batch,totalCount:total,successCount:success,failedCount:failed,failures,status:'processing'};await putBatch(batch)}
    }
    batch={...batch,finishedAt:Date.now(),status:'completed',totalCount:total,successCount:success,failedCount:failed,failures};await putBatch(batch);
    await refreshCount();await renderBatches();setStatus(status,`AI 特征导入完成：成功 ${success} 张，失败 ${failed} 张。同一编号的多张样品图会在查询结果中自动合并。${failed?'可在“导入批次”下载失败报告。':''}`,failed>0&&success===0);
  }catch(e){
    console.error(e);batch={...batch,finishedAt:Date.now(),status:'failed',totalCount:total,successCount:success,failedCount:failed||1,failures:[...failures,{row:null,error:e.message||'文件结构无法识别'}]};await putBatch(batch);await renderBatches();setStatus(status,`Excel 导入失败：${e.message||'文件结构无法识别'}`,true);
  }finally{btn.disabled=false}
}

function bindQuery(input){
  if(!input)return;
  input.onchange=e=>{
    const f=e.target.files[0];
    if(!f)return;
    showPreview(f,$('#queryPreview'));
    handleQuery(f);
    // 允许再次选择同一张图片
    input.value='';
  };
}
async function importBackup(file){
  const data=JSON.parse(await file.text()),items=Array.isArray(data)?data:data.items;if(!Array.isArray(items))throw Error('bad');
  const id=batchId(),now=Date.now(),name=`备份导入：${file.name}`;let success=0;
  const t=db.transaction([SAMPLE_STORE,BATCH_STORE],'readwrite'),samples=t.objectStore(SAMPLE_STORE),batches=t.objectStore(BATCH_STORE);
  items.forEach(x=>{const y=normalizedRecord(x);delete y.id;delete y.location;if(y.localFeatures&&!ArrayBuffer.isView(y.localFeatures))y.localFeatures=Int8Array.from(Array.isArray(y.localFeatures)?y.localFeatures:Object.values(y.localFeatures));y.batchId=id;y.source='backup';y.sourceFile=file.name;samples.add(y);success++});
  batches.put({batchId:id,fileName:name,fileSize:file.size,fingerprint:fileFingerprint(file),startedAt:now,finishedAt:now,status:'completed',totalCount:items.length,successCount:success,failedCount:0,failures:[]});
  await new Promise((r,j)=>{t.oncomplete=r;t.onerror=()=>j(t.error)});return success;
}
function initUI(){
  document.querySelectorAll('.tab').forEach(btn=>btn.onclick=()=>{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));btn.classList.add('active');$('#'+btn.dataset.tab).classList.add('active');if(btn.dataset.tab==='manage')renderBatches()});
  bindQuery($('#queryCameraInput'));bindQuery($('#queryFileInput'));
  $('#stockInput').onchange=e=>{stockFile=e.target.files[0]||null;if(stockFile)showPreview(stockFile,$('#stockPreview'))};
  $('#addForm').onsubmit=async e=>{e.preventDefault();if(!stockFile)return;const btn=$('#saveBtn'),status=$('#addStatus');btn.disabled=true;btn.textContent='正在计算 AI 特征…';try{const {embedding,localFeatures,colorFeature,featureVersion,thumb}=await featuresFromFile(stockFile,status);await addSample({code:$('#shoeCode').value.trim(),customerNo:'',orderNo:'',sendDate:'',remark:'',thumb,embedding,localFeatures,colorFeature,featureVersion,source:'manual',batchId:null,createdAt:Date.now()});setStatus(status,'保存成功；同编号的多张图片会在查询时自动合并。');e.target.reset();stockFile=null;$('#stockPreview').classList.add('hidden');await refreshCount()}catch(err){console.error(err);setStatus(status,`保存失败：${err.message||'浏览器存储空间不足'}`,true)}finally{btn.disabled=false;btn.textContent='计算 AI 特征并保存'}};
  $('#excelInput').onchange=e=>{excelFile=e.target.files[0]||null;$('#excelFileName').textContent=excelFile?`${excelFile.name} · ${(excelFile.size/1024/1024).toFixed(1)} MB`:'';$('#excelImportBtn').disabled=!excelFile};
  $('#excelImportBtn').onclick=()=>excelFile&&importExcel(excelFile);
  $('#exportBtn').onclick=async()=>{const all=(await getAllSamples()).map(x=>({...x,localFeatures:x.localFeatures?Array.from(x.localFeatures):null})),batches=await getAllBatches();downloadJSON(`sample-backup-${new Date().toISOString().slice(0,10)}.json`,{version:5,featureVersion:FEATURE_VERSION,exportedAt:Date.now(),items:all,batches});setStatus($('#manageStatus'),`已导出 ${all.length} 条样品、AI 特征及批次记录。`)};
  $('#importInput').onchange=async e=>{const f=e.target.files[0];if(!f)return;try{const count=await importBackup(f);await refreshCount();await renderBatches();setStatus($('#manageStatus'),`成功导入 ${count} 条备份数据。`)}catch(err){console.error(err);setStatus($('#manageStatus'),'导入失败：备份文件格式不正确。',true)}};
  $('#listBtn').onclick=renderInventory;
  $('#refreshBatchBtn').onclick=renderBatches;
  $('#clearBtn').onclick=async()=>{const input=prompt('此操作会删除全部样品数据，但保留批次审计记录。\n请输入 DELETE 确认：');if(input!=='DELETE'){setStatus($('#manageStatus'),'已取消清空。');return}const samples=await getAllSamples();await clearSamples();const batches=await getAllBatches();for(const b of batches){if(b.status!=='deleted')await markBatchDeleted(b,samples.filter(x=>x.batchId===b.batchId).length)}await refreshCount();await renderBatches();$('#inventoryList').innerHTML='';setStatus($('#manageStatus'),`已清空 ${samples.length} 条样品，导入审计记录已保留。`)};
}
(async()=>{if(!('indexedDB'in window)){alert('当前浏览器不支持本地数据库，请使用最新版 Chrome、Edge 或 Safari。');return}db=await openDB();initUI();await refreshCount();await renderBatches();if('serviceWorker'in navigator&&location.protocol.startsWith('http'))navigator.serviceWorker.register('./sw.js').catch(console.warn)})();

// V3.1 图片预览与文件切换优化
document.addEventListener('click',e=>{
  if(e.target.id==='imageModal'||e.target.id==='modalClose') closeImageModal();
});
document.addEventListener('keydown',e=>{
  if(e.key==='Escape') closeImageModal();
});
// 查询图片预览由 bindQuery 统一处理，避免重复绑定 change 事件。
