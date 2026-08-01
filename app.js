'use strict';

const DB_NAME = 'shoe-match-db';
const DB_VERSION = 1;
const STORE = 'shoes';
let db;
let stockFile = null;

const $ = (s) => document.querySelector(s);
const sleepFrame = () => new Promise(r => requestAnimationFrame(r));

async function openDB(){
  return new Promise((resolve,reject)=>{
    const req=indexedDB.open(DB_NAME,DB_VERSION);
    req.onupgradeneeded=()=>{
      const d=req.result;
      if(!d.objectStoreNames.contains(STORE)){
        const st=d.createObjectStore(STORE,{keyPath:'id',autoIncrement:true});
        st.createIndex('code','code',{unique:false});
        st.createIndex('createdAt','createdAt',{unique:false});
      }
    };
    req.onsuccess=()=>resolve(req.result);
    req.onerror=()=>reject(req.error);
  });
}
function tx(mode='readonly'){return db.transaction(STORE,mode).objectStore(STORE)}
function reqP(req){return new Promise((res,rej)=>{req.onsuccess=()=>res(req.result);req.onerror=()=>rej(req.error)})}
async function getAll(){return reqP(tx().getAll())}
async function addShoe(v){return reqP(tx('readwrite').add(v))}
async function deleteShoe(id){return reqP(tx('readwrite').delete(id))}
async function clearAll(){return reqP(tx('readwrite').clear())}
async function countAll(){return reqP(tx().count())}

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
  const scale=Math.max(size/img.width,size/img.height);
  const w=img.width*scale,h=img.height*scale;
  ctx.drawImage(img,(size-w)/2,(size-h)/2,w,h);
  return c;
}
function makeThumb(img){
  const c=drawCover(img,420);
  return c.toDataURL('image/jpeg',0.78);
}
function grayAt(data,i){return data[i]*0.299+data[i+1]*0.587+data[i+2]*0.114}
function extractFeatures(img){
  // 1) 256-bit dHash
  const hc=document.createElement('canvas'); hc.width=17; hc.height=16;
  const hctx=hc.getContext('2d',{willReadFrequently:true});
  const cover=drawCover(img,256); hctx.drawImage(cover,0,0,17,16);
  const hd=hctx.getImageData(0,0,17,16).data;
  let bits='';
  for(let y=0;y<16;y++) for(let x=0;x<16;x++){
    const a=grayAt(hd,(y*17+x)*4),b=grayAt(hd,(y*17+x+1)*4);
    bits+=a>b?'1':'0';
  }
  let hash='';
  for(let i=0;i<bits.length;i+=4) hash+=parseInt(bits.slice(i,i+4),2).toString(16);

  // 2) 24维 RGB 直方图
  const c=document.createElement('canvas'); c.width=c.height=64;
  const ctx=c.getContext('2d',{willReadFrequently:true}); ctx.drawImage(cover,0,0,64,64);
  const d=ctx.getImageData(0,0,64,64).data;
  const color=new Array(24).fill(0), gray=new Float32Array(64*64);
  for(let p=0,j=0;p<d.length;p+=4,j++){
    color[Math.min(7,d[p]>>5)]++;
    color[8+Math.min(7,d[p+1]>>5)]++;
    color[16+Math.min(7,d[p+2]>>5)]++;
    gray[j]=grayAt(d,p);
  }
  const pixels=64*64; for(let i=0;i<color.length;i++) color[i]/=pixels;

  // 3) 8方向边缘梯度直方图
  const edge=new Array(8).fill(0); let edgeSum=0;
  for(let y=1;y<63;y++) for(let x=1;x<63;x++){
    const i=y*64+x;
    const gx=-gray[i-65]+gray[i-63]-2*gray[i-1]+2*gray[i+1]-gray[i+63]+gray[i+65];
    const gy=-gray[i-65]-2*gray[i-64]-gray[i-63]+gray[i+63]+2*gray[i+64]+gray[i+65];
    const mag=Math.hypot(gx,gy); if(mag<35) continue;
    let angle=Math.atan2(gy,gx)+Math.PI; let bin=Math.floor(angle/(2*Math.PI)*8)%8;
    edge[bin]+=mag; edgeSum+=mag;
  }
  if(edgeSum>0) for(let i=0;i<8;i++) edge[i]/=edgeSum;
  return {hash,color,edge};
}
function popcntHex(a,b){
  let diff=0; const table=[0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4];
  for(let i=0;i<Math.min(a.length,b.length);i++) diff+=table[parseInt(a[i],16)^parseInt(b[i],16)];
  return diff;
}
function cosine(a,b){let dot=0,aa=0,bb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i]}return dot/(Math.sqrt(aa*bb)||1)}
function similarity(q,s){
  const hashSim=1-popcntHex(q.hash,s.hash)/256;
  const colorSim=Math.max(0,cosine(q.color,s.color));
  const edgeSim=Math.max(0,cosine(q.edge,s.edge));
  return Math.max(0,Math.min(1,hashSim*.58+colorSim*.27+edgeSim*.15));
}
async function featuresFromFile(file){const img=await loadImage(file);return {img,features:extractFeatures(img),thumb:makeThumb(img)}}
function showPreview(file,el){el.src=URL.createObjectURL(file);el.classList.remove('hidden')}
async function refreshCount(){$('#dbCount').textContent=`${await countAll()} 双`}
function setStatus(el,text,error=false){el.textContent=text;el.style.color=error?'#b91c1c':'#475569'}

function renderResults(items){
  const root=$('#results'); root.innerHTML='';
  if(!items.length){root.innerHTML='<div class="empty">鞋库暂无数据，请先入库。</div>';return}
  items.forEach((x,i)=>{
    const n=$('#resultTemplate').content.cloneNode(true);
    n.querySelector('.rank').textContent=i+1;
    n.querySelector('.result-img').src=x.thumb;
    n.querySelector('.result-code').textContent=x.code;
    n.querySelector('.result-location').textContent=`位置：${x.location}`;
    const pct=Math.round(x.score*1000)/10;
    n.querySelector('.result-score').textContent=`${pct}%`;
    n.querySelector('.score-bar i').style.width=`${pct}%`;
    root.appendChild(n);
  });
}

async function handleQuery(file){
  const status=$('#queryStatus'); setStatus(status,'正在计算查询指纹…');
  const start=performance.now();
  try{
    const {features}=await featuresFromFile(file);
    const all=await getAll();
    setStatus(status,`正在比对 ${all.length} 条记录…`); await sleepFrame();
    const scored=[];
    for(let i=0;i<all.length;i++){
      const s=all[i]; scored.push({...s,score:similarity(features,s.features)});
      if(i>0&&i%400===0) await sleepFrame();
    }
    scored.sort((a,b)=>b.score-a.score); renderResults(scored.slice(0,5));
    setStatus(status,`完成：共比对 ${all.length} 条，耗时 ${Math.round(performance.now()-start)} ms。`);
  }catch(e){console.error(e);setStatus(status,'识别失败，请换一张照片重试。',true)}
}

async function renderInventory(){
  const root=$('#inventoryList'); root.innerHTML='';
  const all=(await getAll()).sort((a,b)=>b.createdAt-a.createdAt).slice(0,30);
  if(!all.length){root.innerHTML='<div class="empty">暂无入库数据</div>';return}
  all.forEach(x=>{
    const d=document.createElement('article'); d.className='inventory-item';
    d.innerHTML=`<img alt="库存照"><div><h4></h4><p></p></div><button class="mini-danger">删除</button>`;
    d.querySelector('img').src=x.thumb; d.querySelector('h4').textContent=x.code; d.querySelector('p').textContent=x.location;
    d.querySelector('button').onclick=async()=>{if(confirm(`确定删除 ${x.code}？`)){await deleteShoe(x.id);await refreshCount();renderInventory()}};
    root.appendChild(d);
  });
}

function downloadJSON(name,data){
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));a.download=name;a.click();URL.revokeObjectURL(a.href)
}

function initUI(){
  document.querySelectorAll('.tab').forEach(btn=>btn.onclick=()=>{
    document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));
    btn.classList.add('active'); $('#'+btn.dataset.tab).classList.add('active');
  });
  $('#queryInput').onchange=e=>{const f=e.target.files[0];if(f){showPreview(f,$('#queryPreview'));handleQuery(f)}};
  $('#stockInput').onchange=e=>{stockFile=e.target.files[0]||null;if(stockFile)showPreview(stockFile,$('#stockPreview'))};
  $('#addForm').onsubmit=async e=>{
    e.preventDefault(); if(!stockFile)return;
    const btn=$('#saveBtn'),status=$('#addStatus'); btn.disabled=true;btn.textContent='正在处理…';
    try{
      const {features,thumb}=await featuresFromFile(stockFile);
      await addShoe({code:$('#shoeCode').value.trim(),location:$('#shoeLocation').value.trim(),thumb,features,createdAt:Date.now()});
      setStatus(status,'保存成功。'); e.target.reset();stockFile=null;$('#stockPreview').classList.add('hidden');await refreshCount();
    }catch(err){console.error(err);setStatus(status,'保存失败，可能是浏览器存储空间不足。',true)}
    finally{btn.disabled=false;btn.textContent='计算指纹并保存'}
  };
  $('#exportBtn').onclick=async()=>{const all=await getAll();downloadJSON(`shoe-backup-${new Date().toISOString().slice(0,10)}.json`,{version:1,exportedAt:Date.now(),items:all});setStatus($('#manageStatus'),`已导出 ${all.length} 条。`)};
  $('#importInput').onchange=async e=>{
    const f=e.target.files[0];if(!f)return;
    try{const data=JSON.parse(await f.text());const items=Array.isArray(data)?data:data.items;if(!Array.isArray(items))throw Error('bad');
      const st=tx('readwrite');items.forEach(x=>{const y={...x};delete y.id;st.add(y)});await new Promise((r,j)=>{st.transaction.oncomplete=r;st.transaction.onerror=()=>j(st.transaction.error)});
      await refreshCount();setStatus($('#manageStatus'),`成功导入 ${items.length} 条。`);
    }catch(err){setStatus($('#manageStatus'),'导入失败：备份文件格式不正确。',true)}
  };
  $('#listBtn').onclick=renderInventory;
  $('#clearBtn').onclick=async()=>{if(confirm('确定清空全部鞋库数据？此操作无法撤销。')){await clearAll();await refreshCount();$('#inventoryList').innerHTML='';setStatus($('#manageStatus'),'已清空。')}};
}

(async()=>{
  if(!('indexedDB' in window)){alert('当前浏览器不支持本地数据库，请使用最新版 Chrome、Edge 或 Safari。');return}
  db=await openDB();initUI();await refreshCount();
  if('serviceWorker' in navigator && location.protocol.startsWith('http')) navigator.serviceWorker.register('./sw.js').catch(console.warn);
})();
