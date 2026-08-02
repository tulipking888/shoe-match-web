'use strict';

const DB_NAME = 'shoe-match-db';
const DB_VERSION = 3;
const SAMPLE_STORE = 'shoes';
const BATCH_STORE = 'importBatches';
let db;
let stockFile = null;
let excelFile = null;
let queryToken = 0;

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
    img.onload=()=>resolve(img); img.onerror=reject;
    img.src=typeof fileOrUrl==='string'?fileOrUrl:URL.createObjectURL(fileOrUrl);
  });
}
function drawCover(img,size=256){
  const c=document.createElement('canvas'); c.width=c.height=size;
  const ctx=c.getContext('2d',{willReadFrequently:true});
  const scale=Math.max(size/img.width,size/img.height),w=img.width*scale,h=img.height*scale;
  ctx.drawImage(img,(size-w)/2,(size-h)/2,w,h); return c;
}
function makeThumb(img){return drawCover(img,420).toDataURL('image/jpeg',0.76)}
function grayAt(data,i){return data[i]*.299+data[i+1]*.587+data[i+2]*.114}
function extractFeatures(img){
  const hc=document.createElement('canvas');hc.width=17;hc.height=16;
  const hctx=hc.getContext('2d',{willReadFrequently:true});const cover=drawCover(img,256);hctx.drawImage(cover,0,0,17,16);
  const hd=hctx.getImageData(0,0,17,16).data;let bits='';
  for(let y=0;y<16;y++)for(let x=0;x<16;x++)bits+=grayAt(hd,(y*17+x)*4)>grayAt(hd,(y*17+x+1)*4)?'1':'0';
  let hash='';for(let i=0;i<bits.length;i+=4)hash+=parseInt(bits.slice(i,i+4),2).toString(16);
  const c=document.createElement('canvas');c.width=c.height=64;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(cover,0,0,64,64);
  const d=ctx.getImageData(0,0,64,64).data,color=new Array(24).fill(0),gray=new Float32Array(4096);
  for(let p=0,j=0;p<d.length;p+=4,j++){color[Math.min(7,d[p]>>5)]++;color[8+Math.min(7,d[p+1]>>5)]++;color[16+Math.min(7,d[p+2]>>5)]++;gray[j]=grayAt(d,p)}
  for(let i=0;i<24;i++)color[i]/=4096;
  const edge=new Array(8).fill(0);let edgeSum=0;
  for(let y=1;y<63;y++)for(let x=1;x<63;x++){const i=y*64+x,gx=-gray[i-65]+gray[i-63]-2*gray[i-1]+2*gray[i+1]-gray[i+63]+gray[i+65],gy=-gray[i-65]-2*gray[i-64]-gray[i-63]+gray[i+63]+2*gray[i+64]+gray[i+65],mag=Math.hypot(gx,gy);if(mag<35)continue;const bin=Math.floor((Math.atan2(gy,gx)+Math.PI)/(2*Math.PI)*8)%8;edge[bin]+=mag;edgeSum+=mag}
  if(edgeSum)for(let i=0;i<8;i++)edge[i]/=edgeSum;
  return {hash,color,edge};
}
function popcntHex(a,b){let diff=0;const t=[0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4];for(let i=0;i<Math.min(a.length,b.length);i++)diff+=t[parseInt(a[i],16)^parseInt(b[i],16)];return diff}
function cosine(a,b){let dot=0,aa=0,bb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i]}return dot/(Math.sqrt(aa*bb)||1)}
function similarity(q,s){const hs=1-popcntHex(q.hash,s.hash)/256,cs=Math.max(0,cosine(q.color,s.color)),es=Math.max(0,cosine(q.edge,s.edge));return Math.max(0,Math.min(1,hs*.58+cs*.27+es*.15))}
async function featuresFromFile(file){const img=await loadImage(file);return {img,features:extractFeatures(img),thumb:makeThumb(img)}}
function showPreview(file,el){el.src=URL.createObjectURL(file);el.classList.remove('hidden')}
async function refreshCount(){$('#dbCount').textContent=`${await countAll()} 条`;await refreshStorage()}
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
    const pct=Math.round(x.score*1000)/10;n.querySelector('.result-score').textContent=`${pct}%`;n.querySelector('.score-bar i').style.width=`${pct}%`;root.appendChild(n);
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
    const {features}=await featuresFromFile(file);
    if(token!==queryToken)return;
    const all=(await getAllSamples()).filter(x=>x.features);
    setStatus(status,`正在比对 ${all.length} 条记录…`);
    await sleepFrame();
    const scored=[];
    for(let i=0;i<all.length;i++){
      if(token!==queryToken)return;
      const s=normalizedRecord(all[i]);
      scored.push({...s,score:similarity(features,s.features)});
      if(i&&i%300===0)await sleepFrame();
    }
    scored.sort((a,b)=>b.score-a.score);
    if(token!==queryToken)return;
    renderResults(scored.slice(0,5));
    setStatus(status,`完成：比对 ${all.length} 条，耗时 ${Math.round(performance.now()-start)} ms。`);
  }catch(e){
    console.error(e);
    setStatus(status,'识别失败，请换一张图片重试。',true);
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
function pathJoin(base,target){const parts=(base+'/'+target).split('/'),out=[];for(const p of parts){if(!p||p==='.')continue;if(p==='..')out.pop();else out.push(p)}return out.join('/')}
function colToIndex(ref){let n=0;for(const ch of ref.match(/[A-Z]+/i)[0].toUpperCase())n=n*26+ch.charCodeAt(0)-64;return n-1}
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
  const imagesByRow=new Map();const drawingFile=zip.file(drawingPath);
  if(drawingFile){const drawingDoc=xml(await drawingFile.async('text')),relsPath=drawingPath.replace(/([^/]+)$/,'_rels/$1.rels'),relsFile=zip.file(relsPath);if(relsFile){const relsDoc=xml(await relsFile.async('text')),relMap=new Map(localChildren(relsDoc,'Relationship').map(r=>[r.getAttribute('Id'),pathJoin(drawingPath.split('/').slice(0,-1).join('/'),r.getAttribute('Target'))]));
      for(const anchor of [...drawingDoc.documentElement.children]){const from=[...anchor.children].find(n=>n.localName==='from'),pic=localChildren(anchor,'pic')[0];if(!from||!pic)continue;const row0=Number(localChildren(from,'row')[0]?.textContent),blip=localChildren(pic,'blip')[0],rid=blip?.getAttributeNS('http://schemas.openxmlformats.org/officeDocument/2006/relationships','embed')||blip?.getAttribute('r:embed');const mediaPath=relMap.get(rid);if(mediaPath&&!imagesByRow.has(row0+1))imagesByRow.set(row0+1,mediaPath)}
    }}
  return {zip,rows,names,imagesByRow};
}
function pick(obj,names,candidates){for(const name of candidates){const idx=names[name];if(idx!==undefined)return obj[idx]??''}return ''}

async function findDuplicateBatch(file){
  const fp=fileFingerprint(file);return (await getAllBatches()).filter(x=>x.fingerprint===fp&&x.status!=='deleted').sort((a,b)=>b.startedAt-a.startedAt)[0];
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
  const btn=$('#excelImportBtn'),status=$('#excelStatus'),wrap=$('#excelProgressWrap'),bar=$('#excelProgressBar'),txt=$('#excelProgressText');btn.disabled=true;wrap.classList.remove('hidden');bar.style.width='0%';setStatus(status,'正在解析 Excel，请勿关闭页面…');
  let success=0,failed=0,total=0;const failures=[];
  let batch={batchId:id,fileName:file.name,fileSize:file.size,fingerprint,startedAt,status:'processing',totalCount:0,successCount:0,failedCount:0,failures:[]};await putBatch(batch);
  try{
    const {zip,rows,names,imagesByRow}=await parseExcelPackage(file);const candidates=rows.filter(r=>String(pick(r,names,['样品编号','鞋子编号','编号'])).trim()||imagesByRow.has(r._row));total=candidates.length;batch.totalCount=total;await putBatch(batch);
    for(let i=0;i<candidates.length;i++){
      const r=candidates[i],code=String(pick(r,names,['样品编号','鞋子编号','编号'])).trim(),mediaPath=imagesByRow.get(r._row);
      try{
        if(!code)throw Error('样品编号为空');if(!mediaPath)throw Error('缺少内嵌图片');const zf=zip.file(mediaPath);if(!zf)throw Error('找不到图片文件');const blob=await zf.async('blob');const {features,thumb}=await featuresFromFile(blob);
        await addSample({code,sendDate:excelDate(pick(r,names,['样品寄出时间','寄出时间','日期'])),customerNo:String(pick(r,names,['客户编号','客户号'])).trim(),orderNo:String(pick(r,names,['订单编号','订单号'])).trim(),remark:String(pick(r,names,['特别要求','备注','要求'])).trim(),thumb,features,source:'excel',sourceFile:file.name,sourceRow:r._row,batchId:id,createdAt:Date.now()});success++;
      }catch(e){failed++;failures.push({row:r._row,code,error:e.message||String(e)});console.warn('导入行失败',r._row,e)}
      const pct=Math.round((i+1)/candidates.length*100);bar.style.width=`${pct}%`;txt.textContent=`${i+1} / ${candidates.length} · 成功 ${success} · 失败 ${failed}`;
      if(i%3===0){await sleepFrame();await wait(5)}
      if(i%25===0){batch={...batch,totalCount:total,successCount:success,failedCount:failed,failures,status:'processing'};await putBatch(batch)}
    }
    batch={...batch,finishedAt:Date.now(),status:'completed',totalCount:total,successCount:success,failedCount:failed,failures};await putBatch(batch);
    await refreshCount();await renderBatches();setStatus(status,`导入完成：成功 ${success} 条，失败 ${failed} 条。相同编号已全部保留。${failed?'可在“导入批次”下载失败报告。':''}`,failed>0&&success===0);
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
  items.forEach(x=>{const y=normalizedRecord(x);delete y.id;delete y.location;y.batchId=id;y.source='backup';y.sourceFile=file.name;samples.add(y);success++});
  batches.put({batchId:id,fileName:name,fileSize:file.size,fingerprint:fileFingerprint(file),startedAt:now,finishedAt:now,status:'completed',totalCount:items.length,successCount:success,failedCount:0,failures:[]});
  await new Promise((r,j)=>{t.oncomplete=r;t.onerror=()=>j(t.error)});return success;
}
function initUI(){
  document.querySelectorAll('.tab').forEach(btn=>btn.onclick=()=>{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));btn.classList.add('active');$('#'+btn.dataset.tab).classList.add('active');if(btn.dataset.tab==='manage')renderBatches()});
  bindQuery($('#queryCameraInput'));bindQuery($('#queryFileInput'));
  $('#stockInput').onchange=e=>{stockFile=e.target.files[0]||null;if(stockFile)showPreview(stockFile,$('#stockPreview'))};
  $('#addForm').onsubmit=async e=>{e.preventDefault();if(!stockFile)return;const btn=$('#saveBtn'),status=$('#addStatus');btn.disabled=true;btn.textContent='正在处理…';try{const {features,thumb}=await featuresFromFile(stockFile);await addSample({code:$('#shoeCode').value.trim(),customerNo:'',orderNo:'',sendDate:'',remark:'',thumb,features,source:'manual',batchId:null,createdAt:Date.now()});setStatus(status,'保存成功；编号重复也会作为独立记录保留。');e.target.reset();stockFile=null;$('#stockPreview').classList.add('hidden');await refreshCount()}catch(err){console.error(err);setStatus(status,'保存失败，可能是浏览器存储空间不足。',true)}finally{btn.disabled=false;btn.textContent='计算特征并保存'}};
  $('#excelInput').onchange=e=>{excelFile=e.target.files[0]||null;$('#excelFileName').textContent=excelFile?`${excelFile.name} · ${(excelFile.size/1024/1024).toFixed(1)} MB`:'';$('#excelImportBtn').disabled=!excelFile};
  $('#excelImportBtn').onclick=()=>excelFile&&importExcel(excelFile);
  $('#exportBtn').onclick=async()=>{const all=await getAllSamples(),batches=await getAllBatches();downloadJSON(`sample-backup-${new Date().toISOString().slice(0,10)}.json`,{version:3,exportedAt:Date.now(),items:all,batches});setStatus($('#manageStatus'),`已导出 ${all.length} 条样品及批次记录。`)};
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
