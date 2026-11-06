import { createApp } from './apps.js';
import { createComputer } from './machine/lab.js';

const $ = (s, root = document) => root.querySelector(s);
const escape = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paths = {
 computer:'<rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4M19 9h4m-4 6h4"/><path d="M9 9h6v6H9z"/>',
 orbit:'<circle cx="12" cy="12" r="5.5"/><ellipse cx="12" cy="12" rx="11" ry="3.8" transform="rotate(-32 12 12)"/>',
 welcome:'<circle cx="12" cy="12" r="5.5"/><ellipse cx="12" cy="12" rx="11" ry="3.8" transform="rotate(-32 12 12)"/>',
 files:'<path d="M2 7a2 2 0 0 1 2-2h5l2 2h9a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2z" fill="currentColor" stroke="none" opacity=".9"/><path d="M2 11h20" stroke="#78a9dd"/><path d="M4 8V5h5l2 2"/>',
 notes:'<path d="M6 3h12a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1" fill="#fff9e8" stroke="none"/><path d="M9 8h7M9 12h7M9 16h5" stroke-width="1.2"/><path d="M6 3v18" opacity=".25"/>',
 terminal:'<path d="m5 7 5 5-5 5M13 17h6" stroke-width="1.9"/>',
 calculator:'<rect x="4" y="2" width="16" height="20" rx="2.5"/><path d="M7 6h10v4H7z" fill="currentColor" stroke="none"/><path d="M7 14h2m-1-1v2m5-1h3M7 18h2m4 0h3" stroke-width="1.6"/>',
 settings:'<path d="m10 2-.7 2.4-2 .9L5 4.5 2.5 9l1.9 1.7v2.6L2.5 15 5 19.5l2.3-.8 2 .9.7 2.4h4l.7-2.4 2-.9 2.3.8 2.5-4.5-1.9-1.7v-2.6L21.5 9 19 4.5l-2.3.8-2-.9L14 2z" stroke-width="1.5"/><circle cx="12" cy="12" r="3.5"/>',
 launcher:'<rect x="3" y="3" width="6" height="6" rx="1"/><rect x="15" y="3" width="6" height="6" rx="1"/><rect x="3" y="15" width="6" height="6" rx="1"/><rect x="15" y="15" width="6" height="6" rx="1"/>',
 search:'<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
 wifi:'<path d="M2 8a16 16 0 0 1 20 0M5 12a11 11 0 0 1 14 0m-11 4a6 6 0 0 1 8 0"/><circle cx="12" cy="20" r=".7" fill="currentColor"/>',
 fullscreen:'<path d="M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5"/>',
 controls:'<path d="M3 7h18M3 17h18"/><rect x="7" y="4" width="4" height="6" rx="1.5" fill="#dfeadd"/><rect x="14" y="14" width="4" height="6" rx="1.5" fill="#dfeadd"/>'
};
const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.notes}</svg>`;
const appIcon = id => `<span class="app-icon icon-${id}">${icon(id)}</span>`;
const catalog = [
 {id:'computer',name:'Neural Computer',description:'A RISC-V CPU in the weights'},
 {id:'welcome',name:'Welcome',description:'Make yourself at home'},
 {id:'files',name:'Files',description:'A place for everything'},
 {id:'terminal',name:'Terminal',description:'A little command magic'},
 {id:'notes',name:'Notes',description:'Catch a passing thought'},
 {id:'calculator',name:'Calculator',description:'Make things add up'},
 {id:'settings',name:'Settings',description:'Make this space yours'}
];
function readStored(key, fallback) {try {const v=JSON.parse(localStorage.getItem(key));return v ?? fallback;}catch{return fallback;}}
const initialFiles = [
 {id:'welcome',name:'A little hello.txt',folder:'Documents',content:'Welcome to your little corner of the internet.\n\nThis is Orbit OS — a computer-shaped space that lives right in your browser.\n\nA few things to try:\n• Write a thought in Notes. It saves as you type.\n• Make a file in Files, then find it in Terminal.\n• Type “help” in Terminal to see what it can do.\n• Pick a new landscape in Settings.\n• Press ⌘ K (or Ctrl K) to open anything.\n\nYour files stay in this browser. Download anything you want to keep elsewhere.\n\nMake yourself at home.\n— Orbit',updated:new Date().toISOString()},
 {id:'ideas',name:'Small ideas.txt',folder:'Documents',content:'Things I want to make\n\n01. A place for a new idea\n02. Something that makes someone smile\n03. A little more room to think\n\nWhat comes next?',updated:new Date().toISOString()},
 {id:'readme',name:'Read me.txt',folder:'Home',content:'ORBIT OS 1.0\nA local desktop in your browser.\n\nFiles are saved with browser local storage on this device.\nNo server shell or remote machine is connected.\n\nTry: help, ls, cat, touch, mkdir, open, echo, date, neofetch\n\nEnjoy your stay.',updated:new Date().toISOString()}
];
let savedFiles=readStored('orbit.files',initialFiles);
const files = Array.isArray(savedFiles) ? savedFiles : initialFiles;
let wallpaper = readStored('orbit.wallpaper','aurora');
if(!['aurora','dusk','mint'].includes(wallpaper)) wallpaper='aurora';
document.body.dataset.wallpaper=wallpaper;
let zIndex=10, cascade=0, active=null, toastTimer;
const wins = new Map();
function saveFiles(){try{localStorage.setItem('orbit.files',JSON.stringify(files));}catch{toast('Browser storage is full. Download your files to keep them.');}window.dispatchEvent(new CustomEvent('orbit-files-changed'));}
function toast(msg){$('#toast').textContent=msg;$('#toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('visible'),3000);}
function setWallpaper(value){if(!['aurora','dusk','mint'].includes(value))return;wallpaper=value;document.body.dataset.wallpaper=value;try{localStorage.setItem('orbit.wallpaper',JSON.stringify(value));}catch{}toast('A fresh perspective. Wallpaper updated.');}
function updateDock(){for(const el of document.querySelectorAll('[data-dock]')){el.classList.toggle('running',wins.has(el.dataset.dock));}}
function focusWindow(id){const w=wins.get(id);if(!w)return;for(const item of wins.values())item.el.classList.remove('focused');w.el.hidden=false;w.el.style.zIndex=++zIndex;w.el.classList.add('focused');active=id;$('#active-app').textContent=catalog.find(a=>a.id===id)?.name||'Orbit OS';w.onFocus?.();}
function nextFocus(){const next=[...wins.values()].filter(w=>!w.el.hidden).sort((a,b)=>Number(b.el.style.zIndex)-Number(a.el.style.zIndex))[0];if(next)focusWindow(next.id);else{active=null;$('#active-app').textContent='Desktop';}}
function closeWindow(id){const w=wins.get(id);if(!w)return;w.destroy?.();w.el.remove();wins.delete(id);updateDock();nextFocus();}
function maximizeWindow(id){const w=wins.get(id);w?.el.classList.toggle('maximized');focusWindow(id);}
function welcomeApp(){const el=document.createElement('div');el.className='welcome';el.innerHTML=`<aside class="welcome-side"><span class="welcome-eyebrow">MEET YOUR NEW DESKTOP</span><h1>Your space,<br>in orbit.</h1><p>A little computer.<br>A world of possibilities.<br>All right here in your browser.</p><div class="orbit-art" aria-hidden="true"><div class="ring"></div><div class="planet"></div></div><span class="welcome-side-bottom"><i></i> A SMALL SPACE TO DO BIG THINGS</span></aside><div class="welcome-main"><div class="welcome-main-header"><h2>Make yourself at home.</h2><span class="small-tag">ORBIT OS 1.0</span></div><p class="welcome-intro">A neural computer and a place to make it your own.<br>Open the machine to run code or explore the circuit.</p><div class="app-grid">${['computer','notes','terminal','files'].map(id=>{let app=catalog.find(a=>a.id===id);return `<button class="app-card" data-open="${id}">${appIcon(id)}<span><strong>${app.name}</strong><small>${app.description}</small></span><span class="arrow">↗</span></button>`;}).join('')}</div><div class="welcome-tip"><span class="sparkle">✧</span><span><strong>A shortcut to anywhere.</strong><br>Press <kbd>${navigator.platform.includes('Mac')?'⌘':'ctrl'}</kbd> + <kbd>K</kbd> to find an app or file in a flash.</span></div><div class="welcome-bottom"><span>Made for the curious. Built to feel like home.</span><button data-open="settings">Make it yours <span>↗</span></button></div></div>`;el.querySelectorAll('[data-open]').forEach(b=>b.onclick=()=>openApp(b.dataset.open));return {element:el};}
function openApp(id,options={}){
 const app=catalog.find(a=>a.id===id);if(!app)return;
 closeLauncher();
 if(wins.has(id)){
  if(id==='notes'&&options.fileId){closeWindow(id);}else{focusWindow(id);return;}
 }
 const api={files,saveFiles,openApp,toast,setWallpaper,get wallpaper(){return wallpaper;},escape,appOptions:options};
 let content;try{content=id==='welcome'?welcomeApp():id==='computer'?createComputer(api):createApp(id,api);}catch(error){console.error(error);toast('This app could not open. Please refresh and try again.');return;}
 const el=document.createElement('section');el.className='window';el.setAttribute('aria-label',app.name+' window');el.setAttribute('role','region');
 const desktopWidth=window.innerWidth, desktopHeight=window.innerHeight-145;
 const sizes={computer:[1040,650],welcome:[790,482],files:[760,490],terminal:[705,445],notes:[730,490],calculator:[350,490],settings:[680,480]};
 const [desiredW,desiredH]=sizes[id];let width=Math.min(desiredW,desktopWidth-130),height=Math.min(desiredH,desktopHeight-30);width=Math.max(320,width);height=Math.max(260,height);
 let left=Math.max(110,(desktopWidth-width)/2-26+(cascade%4)*22),top=Math.max(28,(desktopHeight-height)/2+8+(cascade%4)*19);cascade++;
 if(desktopWidth<900){left=Math.max(15,(desktopWidth-width)/2);top=Math.max(80,(desktopHeight-height)/2);}
 el.style.cssText=`width:${width}px;height:${height}px;left:${Math.min(left,desktopWidth-width-10)}px;top:${top}px;`;
 el.innerHTML=`<header class="titlebar"><div class="window-controls"><button class="control-close" title="Close ${app.name}" aria-label="Close ${app.name}">×</button><button class="control-min" title="Minimize ${app.name}" aria-label="Minimize ${app.name}">−</button><button class="control-max" title="Maximize ${app.name}" aria-label="Maximize ${app.name}">+</button></div><span class="window-title">${escape(content.title||app.name)}</span><span class="titlebar-end">ORBIT</span></header><div class="window-content"></div><div class="resize-handle" aria-hidden="true"></div>`;
 $('.window-content',el).append(content.element);$('#windows').append(el);
 wins.set(id,{id,el,destroy:content.destroy,onFocus:content.onFocus});
 $('.control-close',el).onclick=()=>closeWindow(id);
 $('.control-min',el).onclick=()=>{el.hidden=true;nextFocus();};
 $('.control-max',el).onclick=()=>maximizeWindow(id);
 el.addEventListener('pointerdown',()=>{if(active!==id)focusWindow(id);});
 $('.titlebar',el).addEventListener('dblclick',e=>{if(!e.target.closest('button'))maximizeWindow(id);});
 const drag=$('.titlebar',el);drag.addEventListener('pointerdown',e=>{
  if(e.target.closest('button')||el.classList.contains('maximized')||window.innerWidth<=600)return;
  e.preventDefault();focusWindow(id);const startX=e.clientX,startY=e.clientY,startL=el.offsetLeft,startT=el.offsetTop;drag.setPointerCapture(e.pointerId);
  const move=evt=>{el.style.left=Math.min(window.innerWidth-100,Math.max(0,startL+evt.clientX-startX))+'px';el.style.top=Math.min(window.innerHeight-190,Math.max(0,startT+evt.clientY-startY))+'px';};
  const stop=()=>{drag.removeEventListener('pointermove',move);drag.removeEventListener('pointerup',stop);drag.removeEventListener('pointercancel',stop);};drag.addEventListener('pointermove',move);drag.addEventListener('pointerup',stop);drag.addEventListener('pointercancel',stop);
 });
 const resize=$('.resize-handle',el);resize.addEventListener('pointerdown',e=>{e.preventDefault();const x=e.clientX,y=e.clientY,w=el.offsetWidth,h=el.offsetHeight;resize.setPointerCapture(e.pointerId);const move=ev=>{el.style.width=Math.min(window.innerWidth-el.offsetLeft-10,Math.max(320,w+ev.clientX-x))+'px';el.style.height=Math.min(window.innerHeight-el.offsetTop-155,Math.max(240,h+ev.clientY-y))+'px';};const stop=()=>{resize.removeEventListener('pointermove',move);resize.removeEventListener('pointerup',stop);resize.removeEventListener('pointercancel',stop);};resize.addEventListener('pointermove',move);resize.addEventListener('pointerup',stop);resize.addEventListener('pointercancel',stop);});
 updateDock();focusWindow(id);
}
for(const id of ['computer','files','notes']){const app=catalog.find(a=>a.id===id);const b=document.createElement('button');b.className='desktop-shortcut';b.innerHTML=appIcon(id)+`<span>${app.name}</span>`;b.onclick=()=>openApp(id);$('.desktop-shortcuts').append(b);}
$('#dock').innerHTML=`<button class="dock-item" id="dock-launcher" aria-label="Open app launcher">${appIcon('launcher')}<span class="dock-label">Launchpad</span><span class="dock-dot"></span></button><span class="dock-separator"></span>${catalog.map(app=>`<button class="dock-item" data-dock="${app.id}" aria-label="Open ${app.name}">${appIcon(app.id)}<span class="dock-label">${app.name}</span><span class="dock-dot"></span></button>`).join('')}`;
for(const b of document.querySelectorAll('[data-dock]'))b.onclick=()=>{const w=wins.get(b.dataset.dock);if(w&&active===b.dataset.dock&&!w.el.hidden){w.el.hidden=true;nextFocus();}else openApp(b.dataset.dock);};
let results=[],selected=0;
function closeLauncher(){$('#launcher-overlay').hidden=true;}
function renderResults(){const q=$('#launcher-search').value.toLowerCase().trim();const apps=catalog.filter(a=>(a.name+' '+a.description).toLowerCase().includes(q)).map(a=>({...a,type:'app'}));const matchingFiles=q?files.filter(f=>f.name.toLowerCase().includes(q)).map(f=>({id:f.id,name:f.name,description:f.folder||'Home',type:'file'})):[];results=[...apps,...matchingFiles].slice(0,12);selected=Math.min(selected,Math.max(0,results.length-1));$('#launcher-results').innerHTML=results.length?results.map((a,i)=>`<button class="launch-result ${i===selected?'selected':''}" data-result="${i}">${appIcon(a.type==='file'?'notes':a.id)}<span><strong>${escape(a.name)}</strong><small>${escape(a.description)}</small></span><span>${a.type==='file'?'File':'App'} ↗</span></button>`).join(''):'<p style="padding:20px;color:#7b8a71">No apps or files found. Try another search.</p>';for(const b of document.querySelectorAll('[data-result]'))b.onclick=()=>launchResult(Number(b.dataset.result));}
function launchResult(index){const r=results[index];if(!r)return;if(r.type==='file')openApp('notes',{fileId:r.id});else openApp(r.id);}
function toggleLauncher(){const overlay=$('#launcher-overlay');overlay.hidden=!overlay.hidden;if(!overlay.hidden){$('#launcher-search').value='';selected=0;renderResults();$('#launcher-search').focus();}}
$('#dock-launcher').onclick=toggleLauncher;$('#orbit-menu').onclick=toggleLauncher;$('#orbit-menu').innerHTML=icon('orbit');$('#search-icon').innerHTML=icon('search');$('#wifi-icon').innerHTML=icon('wifi');$('#fullscreen').innerHTML=icon('fullscreen');$('#control-button').innerHTML=icon('controls');
$('#control-button').onclick=()=>openApp('settings');$('#help-button').onclick=()=>openApp('notes',{fileId:'welcome'});$('#clock-button').onclick=()=>toast(new Intl.DateTimeFormat(undefined,{dateStyle:'full',timeStyle:'short'}).format(new Date()));
$('#fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{toast('Fullscreen is unavailable in this browser.');}};
$('#launcher-overlay').onclick=e=>{if(e.target===$('#launcher-overlay'))closeLauncher();};
$('#launcher-search').addEventListener('input',()=>{selected=0;renderResults();});
$('#launcher-search').addEventListener('keydown',e=>{if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();selected=(selected+(e.key==='ArrowDown'?1:-1)+results.length)%Math.max(1,results.length);renderResults();}if(e.key==='Enter')launchResult(selected);if(e.key==='Tab'){e.preventDefault();selected=(selected+(e.shiftKey?-1:1)+results.length)%Math.max(1,results.length);renderResults();}});
document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();toggleLauncher();}if(e.key==='Escape'){closeLauncher();$('#context-menu').hidden=true;}if(e.altKey&&e.key==='Tab'){e.preventDefault();const visible=[...wins.keys()].filter(id=>!wins.get(id).el.hidden);const idx=visible.indexOf(active);if(visible.length)focusWindow(visible[(idx+1)%visible.length]);}});
$('#desktop').addEventListener('contextmenu',e=>{if(e.target.closest('.window'))return;e.preventDefault();const menu=$('#context-menu');menu.hidden=false;menu.style.left=Math.min(e.clientX,window.innerWidth-215)+'px';menu.style.top=Math.min(e.clientY,window.innerHeight-125)+'px';});
for(const b of document.querySelectorAll('[data-context]'))b.onclick=()=>{openApp(b.dataset.context);$('#context-menu').hidden=true;};document.addEventListener('pointerdown',e=>{if(!e.target.closest('#context-menu'))$('#context-menu').hidden=true;});
function updateClock(){const now=new Date();$('#clock-button').textContent=(window.innerWidth<=600?'':new Intl.DateTimeFormat('en-US',{weekday:'short',month:'short',day:'numeric'}).format(now)+'  ·  ')+new Intl.DateTimeFormat('en-US',{hour:'numeric',minute:'2-digit'}).format(now);$('#day-label').textContent=new Intl.DateTimeFormat('en-US',{weekday:'long'}).format(now);$('#date-label').textContent=new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric'}).format(now);}
updateClock();setInterval(updateClock,15000);openApp('computer');
window.addEventListener('resize',()=>{if(window.innerWidth<=600)return;for(const w of wins.values()){if(w.el.classList.contains('maximized'))continue;const left=Math.max(10,Math.min(w.el.offsetLeft,window.innerWidth-200)),top=Math.max(0,Math.min(w.el.offsetTop,window.innerHeight-220));w.el.style.left=left+'px';w.el.style.top=top+'px';w.el.style.width=Math.min(w.el.offsetWidth,window.innerWidth-20)+'px';}});
