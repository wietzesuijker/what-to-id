(function(){
'use strict';
function $(id){return document.getElementById(id);}
var css=getComputedStyle(document.documentElement);
function tok(n){return css.getPropertyValue(n).trim();}
var dark=matchMedia('(prefers-color-scheme: dark)').matches,DOT=tok('--dot').split(',').map(Number);
var DAY=864e5,D0=Date.parse(META.day0+'T00:00:00Z'),NO=65535,NOTAX=65535,LAST=META.days-1;
var MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
var MONTH=['January','February','March','April','May','June','July','August','September',
  'October','November','December'];
var FLAG={introduced:1,threatened:2,obscured:4,imprecise:8,earlyObs:16,earlyUp:32},ONLYFLAG={introduced:'introduced',
  threatened:'threatened',exact:'obscured'},ONLYKEYS=Object.keys(ONLYFLAG),nf=new Intl.NumberFormat('en-CA');
function iso(d){return new Date(D0+d*DAY).toISOString().slice(0,10);}
function dayOf(s){var t=Date.parse(s+'T00:00:00Z');return isNaN(t)?null:Math.round((t-D0)/DAY);}
function clamp(d){return Math.max(0,Math.min(LAST,d));}
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){
  return '&#'+c.charCodeAt(0)+';';});}

// The subtitle names the build date; say how old that is, and warn once it is stale.
var STALE_DAYS=2;
(function(){
  var el=$('updated'),t=el?Date.parse(el.getAttribute('datetime')+'T00:00:00Z'):NaN;
  if(isNaN(t))return;
  var n=new Date(),days=Math.floor((Date.UTC(n.getUTCFullYear(),n.getUTCMonth(),n.getUTCDate())-t)/DAY),
    before=el.previousSibling,after=el.nextSibling;
  if(!before||!after||before.nodeType!==3||after.nodeType!==3)return;
  if(days>STALE_DAYS){before.nodeValue='Last updated ';
    el.textContent=days+' days ago';after.nodeValue=after.nodeValue.replace(/^\./,'; some records may have an ID by now.');}
  else{before.nodeValue=before.nodeValue.replace(/ on $/,', updated ');
    el.textContent=days<=0?'today':days===1?'yesterday':days+' days ago';}
})();

// Calendar lookups by day: month of year, day of month, weekday (Monday is 0) and month number
// counted from the first day's month, which indexes META.totals.
var MOY=new Uint8Array(META.days),DOM=new Uint8Array(META.days),DOW=new Uint8Array(META.days),
  MI=new Uint16Array(META.days),Y0=new Date(D0).getUTCFullYear(),M0=new Date(D0).getUTCMonth();
for(var d=0;d<META.days;d++){var t=new Date(D0+d*DAY);MOY[d]=t.getUTCMonth();DOM[d]=t.getUTCDate();
  DOW[d]=(t.getUTCDay()+6)%7;MI[d]=(t.getUTCFullYear()-Y0)*12+MOY[d]-M0;}

// What the viewer picked. Written to the URL hash so a view can be shared or bookmarked.
// Groups and show only hold chip states (see map_chips.js): {Insecta:'inc', Aves:'not'}.
var S={up:false,lo:0,hi:LAST,months:0,groups:{},only:{},picks:[],split:[],rec:null,at:null,fly:false};
(function readHash(){
  var h=new URLSearchParams(location.hash.slice(1));
  S.up=h.get('by')==='uploaded';
  var f=dayOf(h.get('from')||''),t=dayOf(h.get('to')||'');
  if(f!=null)S.lo=clamp(f);if(t!=null)S.hi=clamp(t);
  if(S.lo>S.hi){var sw=S.lo;S.lo=S.hi;S.hi=sw;}
  (h.get('months')||'').split(',').forEach(function(m){if(+m>=1&&+m<=12)S.months|=1<<(m-1);});
  S.groups=CH.read(h.get('groups'),META.groups);
  S.only=CH.read(h.get('only'),ONLYKEYS.filter(function(k){return META.flags.indexOf(ONLYFLAG[k])>=0;}),['exact']);
  S.picks=TX.read(h.get('taxa'));S.oldq=S.picks.length?'':h.get('q')||'';S.rec=+h.get('record')||null;
  S.area=h.get('area')||'';S.park=h.get('park')||'';
  S.bcpark=h.get('bcpark')||'';
  var at=(h.get('at')||'').split('/').map(Number);if(at.length===3&&at.every(isFinite))S.at=at;
  S.fly=!!S.rec&&!S.at;
})();
var PJ=projectFilter({meta:META,hash:new URLSearchParams(location.hash.slice(1)),changed:function(){refilter();}});
var hashTimer=0;
// The selection: the hash without the open record and the view.
function sel(){
  var h=[];if(S.up)h.push('by=uploaded');
  if(S.lo>0)h.push('from='+iso(S.lo));if(S.hi<LAST)h.push('to='+iso(S.hi));
  if(S.months){var ms=[];for(var m=0;m<12;m++)if(S.months>>m&1)ms.push(m+1);h.push('months='+ms);}
  if(!allGroups())h.push('groups='+CH.write(S.groups,META.groups));
  if(CH.any(S.only))h.push('only='+CH.write(S.only,ONLYKEYS));
  if(S.picks.length)h.push('taxa='+TX.write(S.picks));
  if(AREA.hash())h.push(AREA.hash());if(PJ.hash())h.push(PJ.hash());
  return h;}
function writeHash(){clearTimeout(hashTimer);hashTimer=setTimeout(function(){
  var h=sel();
  if(P.n&&SEL>=0)h.push('record='+P.id[SEL]);else if(S.rec)h.push('record='+S.rec);
  var c=map.getCenter();h.push('at='+map.getZoom().toFixed(1)+'/'+c.lat.toFixed(3)+'/'+
    c.lng.toFixed(3));
  history.replaceState(null,'','#'+h.join('&'));},250);}

// Records, concatenated as shards arrive.
var P={n:0,id:new Uint32Array(0),pos:new Float32Array(0),obs:new Uint16Array(0),
  up:new Uint16Array(0),tax:new Uint16Array(0),grp:new Uint8Array(0),ids:new Uint8Array(0),
  fl:new Uint8Array(0)};
var TAXA=null,TF=TX.filter(null,[]),GONE={},GONE_NEW=false,KEEP=new Uint8Array(0),SEL=-1,
  CUM=new Float64Array(META.days+1),TOT=META.totals||null,TC=null,NODAY=0,PER=null,ALT=0;
function cat(a,b){var c=new a.constructor(a.length+b.length);c.set(a);c.set(b,a.length);return c;}
function unzip(buf){
  var u=new Uint8Array(buf,0,Math.min(2,buf.byteLength));
  if(u[0]!==0x1f||u[1]!==0x8b)return Promise.resolve(buf);
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip')))
    .arrayBuffer();
}
function get(url){return fetch(url).then(function(r){
  if(!r.ok)throw new Error('HTTP '+r.status);return r.arrayBuffer();}).then(unzip);}
// A shard stores each wider column byte by byte (every value's low byte, then the next), and ids
// as steps from the previous id.
function addShard(buf,n){
  if(buf.byteLength!==17*n)throw new Error('unexpected size');
  var b=META.bbox,sx=(b[2]-b[0])/65535,sy=(b[3]-b[1])/65535,o=0,u=new Uint8Array(buf);
  function col(T,w){if(w===1){o+=n;return u.slice(o-n,o);}
    var a=new T(n),v=new Uint8Array(a.buffer);
    for(var j=0;j<w;j++,o+=n)for(var k=0;k<n;k++)v[k*w+j]=u[o+k];return a;}
  var id=col(Uint32Array,4),qx=col(Uint16Array,2),qy=col(Uint16Array,2),obs=col(Uint16Array,2),
    up=col(Uint16Array,2),tax=col(Uint16Array,2),grp=col(Uint8Array,1),ids=col(Uint8Array,1),
    fl=col(Uint8Array,1),pos=new Float32Array(2*n);
  for(var s=0,i=0;i<n;i++)id[i]=s=(s+id[i])>>>0;
  for(i=0;i<n;i++){pos[2*i]=b[0]+qx[i]*sx;pos[2*i+1]=b[1]+qy[i]*sy;}
  P={n:P.n+n,id:cat(P.id,id),pos:cat(P.pos,pos),obs:cat(P.obs,obs),up:cat(P.up,up),
    tax:cat(P.tax,tax),grp:cat(P.grp,grp),ids:cat(P.ids,ids),fl:cat(P.fl,fl)};
}

// The map filters on the GPU. Each record carries its observed and uploaded day (-1 for none) as
// filter values, and four categories: group, observed month, uploaded month (12 for no date) and a
// status value, bit 1 introduced, 2 threatened, 4 obscured or imprecise, 8 kept by the taxon picks
// and the area and not deleted, 16 one record in ten. Group, month, status and date changes then
// only change the filter settings; the data is uploaded again only when records arrive, the picks,
// the area or deletions change, or the groups change while a taxon is included (the groups and
// included taxa are one set, so the kept bit then carries the groups too).
var ST_INTRO=1,ST_THREAT=2,ST_INEXACT=4,ST_KEEP=8,ST_SAMPLE=16,PD=null,PD_KEY=null,PD_GONE=-1,PD_CAT=0,
  OUT=new Uint8Array(0),RC=null,PJW=null;
function inc(){return !!TF.inc;}
function pointData(){
  PJ.sync(P);
  var n=P.n,gone=Object.keys(GONE).length,i,key=[TF,inc()?CH.picked(S.groups,META.groups,'inc').join():'',AREA.ver(),PJ.key()];
  if(PD&&PD.length===n&&PD_KEY&&PD_KEY.every(function(k,j){return k===key[j];})&&PD_GONE===gone)return;
  var same=PD&&PD.length===n,days=same?PD.days:new Float32Array(2*n),
    cat=same?PD.attributes.getFilterCategory.value.slice():new Uint8Array(4*n);
  if(!same)for(i=0;i<n;i++){var o=P.obs[i],u=P.up[i],f=P.fl[i];
    days[2*i]=o===NO?-1:o;days[2*i+1]=u===NO?-1:u;
    cat[4*i]=P.grp[i];cat[4*i+1]=o===NO?12:MOY[o];cat[4*i+2]=u===NO?12:MOY[u];
    cat[4*i+3]=(f&FLAG.introduced?ST_INTRO:0)|(f&FLAG.threatened?ST_THREAT:0)|
      (f&(FLAG.obscured|FLAG.imprecise)?ST_INEXACT:0)|(i%10===0?ST_SAMPLE:0);}
  // only the kept bit follows the picks, the groups beside them, the area, projects and deletions
  // OUT keeps the records out for any reason but the area, for the Identify box count; PJW marks
  // the records Identify opens when it cannot say the projects exactly (see map_projects.js)
  var ti=TF.inc,te=TF.exc,gs=key[1]?META.groups.map(function(g){return S.groups[g]==='inc';}):null,tax=P.tax,
    grp=P.grp,id=P.id,nt=NOTAX,am=AREA.mask(n),wide=PJ.wide();
  if(OUT.length!==n)OUT=new Uint8Array(n);PJW=wide?new Uint8Array(n):null;
  for(i=0;i<n;i++){var t=tax[i],keep=!ti||(t!==nt&&ti[t]===1)||(gs!==null&&gs[grp[i]]);
    if(te&&t!==nt&&te[t]===1||gone>0&&GONE[id[i]]===1)keep=false;
    if(wide&&keep&&(am===null||am[i]))PJW[i]=wide(i)?1:0;if(keep&&PJ.drop(i))keep=false;
    OUT[i]=keep?0:1;cat[4*i+3]=cat[4*i+3]&~ST_KEEP|(keep&&(am===null||am[i])?ST_KEEP:0);}
  // a new pick, area or deletion swaps only the categories, so deck.gl uploads only those
  if(same){PD.attributes.getFilterCategory={value:cat,size:4};PD_CAT++;}
  else PD={length:n,days:days,attributes:{getPosition:{value:P.pos,size:2},
    getFilterValue:{value:days,size:2},getFilterCategory:{value:cat,size:4}}};
  PD_KEY=key;PD_GONE=gone;
}
function categories(thin){
  var groups=[],mon=[],all=[],st=[],req=0,hide=0,i;
  CH.groupPass(S.groups,META.groups,inc()).forEach(function(v,j){if(v)groups.push(j);});
  for(i=0;i<13;i++){all.push(i);if(S.months?i<12&&S.months>>i&1:true)mon.push(i);}
  [['introduced',ST_INTRO],['threatened',ST_THREAT],['exact',ST_INEXACT]].forEach(function(o){var v=S.only[o[0]];
    if(v==='not'||v&&o[0]==='exact')hide|=o[1];else if(v)req|=o[1];});
  for(i=0;i<2*ST_SAMPLE;i++)if(i&ST_KEEP&&(i&req)===req&&!(i&hide)&&(!thin||i&ST_SAMPLE))st.push(i);
  return [groups,S.up?all:mon,S.up?mon:all,st];
}
// The counts behind the histograms, header and share run on the CPU after the map has redrawn.
var CT=0;
function refilter(){
  pointData();layers();clearTimeout(CT);CT=setTimeout(recount,0);
}
function recount(){
  var n=P.n,day=S.up?P.up:P.obs,c=categories(),gm=0,mm=0,sm=0,nd=0,cat=PD.attributes.getFilterCategory.value,
    ch=S.up?2:1,g=CH.groupPass(S.groups,META.groups,false),m=S.months;
  c[0].forEach(function(j){gm|=1<<j;});c[ch].forEach(function(j){mm|=1<<j;});c[3].forEach(function(j){sm|=1<<j;});
  RC={gm:gm,mm:mm,sm:sm,ch:ch};
  if(KEEP.length!==n)KEEP=new Uint8Array(n);
  if(!PER)PER=new Float64Array(META.days);else PER.fill(0);
  for(var i=0;i<n;i++){
    var k=(gm>>cat[4*i]&1)&(mm>>cat[4*i+ch]&1)&(sm>>cat[4*i+3]&1);KEEP[i]=k;
    if(k){var d=day[i];if(d!==NO)PER[d]++;else nd++;}
  }
  NODAY=nd;ALT=0;
  if(PJW)for(i=0;i<n;i++)if(PJW[i]&&(gm>>cat[4*i]&1)&(mm>>cat[4*i+ch]&1)&(sm>>(cat[4*i+3]|ST_KEEP)&1)&&
    (day[i]===NO?full():day[i]>=S.lo&&day[i]<=S.hi))ALT++;
  CUM=new Float64Array(META.days+1);
  for(var d=0;d<META.days;d++)CUM[d+1]=CUM[d]+PER[d];
  // One picked status filter reads its own totals; two together, a left-out one, or an area, have none
  var on=Object.keys(S.only),
    tt=!TOT||S.picks.length||AREA.on()||PJ.active()||on.length>1||S.only[on[0]]==='not'?null:
      on.length?TOT.only&&TOT.only[on[0]]:TOT;
  TC=null;
  if(tt){var rows=tt[S.up?'up':'obs'],nm=rows[0].length;TC=new Float64Array(nm+1);
    for(var k=0;k<nm;k++){var v=0;if(!m||m>>(M0+k)%12&1)for(var j=0;j<rows.length;j++)if(g[j])v+=rows[j][k];
      TC[k+1]=TC[k]+v;}}
  if(SEL>=0&&!KEEP[SEL])closeCard();
  render();
}
function full(){return S.lo===0&&S.hi===LAST;}
function render(){
  // records without a date show only over all dates, as the GPU filter does
  var shown=CUM[S.hi+1]-CUM[S.lo]+(full()?NODAY:0);
  $('count').textContent=P.n?nf.format(shown)+' of '+nf.format(META.n)+' records':'Loading';
  var all=P.n&&!$('more').textContent?total(S.lo,S.hi):null;
  $('share').textContent=all?nf.format(Math.min(shown,all))+' of '+nf.format(all)+
    ' BC records with photos '+(full()&&!S.months&&allGroups()&&!CH.any(S.only)?
    '':'matching these filters ')+'still need an ID ('+
    pct(shown,all)+'). All-record counts from '+TOT.on+'.':'';
  $('hint').textContent=TC?HINT_TOT:HINT;
  $('from').value=iso(S.lo);$('to').value=iso(S.hi);
  $('byObs').setAttribute('aria-pressed',String(!S.up));
  $('byUp').setAttribute('aria-pressed',String(S.up));
  PRESETS.forEach(function(p){var r=p.range();
    p.el.setAttribute('aria-pressed',String(r[0]===S.lo&&r[1]===S.hi));});
  PJ.note(P.n?shown:null,ALT);BAR.set(items());tally();draw();layers();link();writeHash();
}
// Whether record i matches the filters with the area left out, as recount() reads them
function match(i){if(!RC||i>=OUT.length)return 0;var c=PD.attributes.getFilterCategory.value,s=c[4*i+3]&~ST_KEEP|(OUT[i]?0:ST_KEEP);
  return (RC.gm>>c[4*i]&1)&(RC.mm>>c[4*i+RC.ch]&1)&(RC.sm>>s&1);}
function link(){
  var b=map.getBounds(),m=[],u=null;for(var i=0;i<12;i++)if(S.months>>i&1)m.push(i+1);
  // with a taxon included, the picked groups go in as their taxa, which iNaturalist adds to the picks;
  // left-out groups go in as taxa to leave out (see CH.groupLink)
  var gl=CH.groupLink(S.groups,META.groups,META.group_ids||{},TF.ids.length>0),
    taxa=gl.inc.length?null:TF.ids.concat(gl.taxa);
  var st={groups:gl.groups,
    all:META.groups.length,up:S.up,d1:S.lo>0?iso(S.lo):'',d2:S.hi<LAST?iso(S.hi):'',months:m,taxa:taxa,
    not:TF.not.concat(gl.not),only:S.only};
  // with an area on, the link may list the records by id instead, and then u stays null
  var a=$('identify');a.href=AREA.link({w:b.getWest(),s:b.getSouth(),e:b.getEast(),n:b.getNorth()},
    function(v,p,side){var r=TX.identifyUrl(st,v,META,Object.assign({},PJ.params(),p));if(!side)u=r;return r.url;},match);
  a.title='Opens these records in the iNaturalist Identify page.';
  // say when Identify cannot show the same records, and what it opens instead
  var why=[],lost=u?u.lost:[];function gn(gs){return gs.map(gname).join(' and ');}
  if(u&&gl.inc.length)why.push('Identify cannot add the '+gn(gl.inc)+' group to picked taxa, so it opens every taxon');
  if(u&&gl.out.length)why.push('Identify cannot leave out the '+gn(gl.out)+' group beside picked taxa, so it keeps it');
  if(lost.indexOf('taxa')>=0)why.push('too many taxa for one link, so it opens every taxon');
  if(lost.indexOf('not')>=0)why.push('too many taxa left out for one link, so it keeps them');
  if(u&&PJ.inexact())why.push(PJ.inexact());
  var nt=$('idnote'),pf=u?PJ.uncounted():'';nt.hidden=!why.length&&!pf;nt.textContent=(why.length?'Identify shows '+
    'more records than the map: '+why.join('; ')+'. ':'')+pf;
}
// The filter extension is kept between calls, as the point data is in pointData(), so deck.gl
// uploads nothing when only the filter settings change. deck.gl would also run every category
// through a lookup in JS and upload the result as 32-bit values; ours are already small numbers,
// so the lookup is the identity and the bytes go up as they are.
class Filter extends deck.DataFilterExtension{
  static componentName='Filter';
  initializeState(context,extension){super.initializeState(context,extension);
    var a=this.getAttributeManager().attributes.filterCategoryValues;if(a)a.settings.transform=null;}
  _getCategoryKey(category){return category;}
}
var EXT=null;
function layers(){
  if(!P.n||!PD)return;
  EXT=EXT||new Filter({filterSize:2,categorySize:4});
  var r=[full()?-1:S.lo,S.hi],any=[-1,LAST];
  var L=[new deck.ScatterplotLayer({id:'pts',data:PD,
    getFillColor:[DOT[0],DOT[1],DOT[2],110],radiusUnits:'pixels',getRadius:1.5,radiusMinPixels:1,
    radiusMaxPixels:6,stroked:false,pickable:true,
    extensions:[EXT],updateTriggers:{getFilterCategory:PD_CAT},filterRange:[S.up?any:r,S.up?r:any],filterCategories:categories(THIN)})];
  if(SEL>=0)L.push(new deck.ScatterplotLayer({id:'sel',data:[SEL],
    getPosition:function(i){return [P.pos[2*i],P.pos[2*i+1]];},radiusUnits:'pixels',getRadius:8,
    filled:false,stroked:true,lineWidthUnits:'pixels',getLineWidth:2.5,
    getLineColor:dark?[255,255,255]:[29,29,27]}));
  overlay.setProps({layers:L,onHover:tip,onClick:pick});
}

// Two histograms. The overview has one bar per year over every date, on a log scale so that sparse
// early years still show. The detail shows the period in view, with bars sized to it (days, weeks,
// months or years) and scaled to its tallest bar. Picking a period in either zooms the detail to it.
var hist=$('hist'),over=$('over'),htip=$('htip'),VIEW=[0,LAST],HE=[0,LAST+1],HU='year',drag=null;
var OE=edges('year',0,LAST);
function count(a,b){return CUM[b+1]-CUM[a];}
// All records, needing an ID or Research Grade, for days a..b under the group and month filters.
// Null when the totals are off or the days do not span whole months (the last day ends a month).
function total(a,b){
  if(!TC||!(a===0||DOM[a]===1)||!(b===LAST||DOM[b+1]===1))return null;
  return TC[MI[b]+1]-TC[MI[a]];
}
function pct(v,t){var f=Math.min(1,v/t)*100;return (f<1&&f>0?'<1':Math.round(f))+'%';}
function viewOf(){
  if(S.lo===0&&S.hi===LAST)return [META.hist0,LAST];
  var pad=Math.max(0,Math.ceil((31-(S.hi-S.lo+1))/2));
  return [clamp(S.lo-pad),clamp(S.hi+pad)];
}
function unitOf(v){var n=v[1]-v[0]+1;return n<=92?'day':n<=732?'week':n<=25*366?'month':'year';}
function edges(u,v0,v1){
  var e=[v0];
  for(var d=v0+1;d<=v1;d++)if(u==='day'||(u==='week'&&DOW[d]===0)||(DOM[d]===1&&
    (u==='month'||(u==='year'&&MOY[d]===0))))e.push(d);
  e.push(v1+1);return e;
}
function year(d){return iso(d).slice(0,4);}
function dlabel(d){return DOM[d]+' '+MON[MOY[d]]+' '+year(d);}
function binLabel(u,a){return u==='day'?dlabel(a):u==='week'?'Week of '+dlabel(a):
  u==='month'?MONTH[MOY[a]]+' '+year(a):year(a);}
function on(a,b){return b>=S.lo&&a<=S.hi&&(!S.months||b-a>40||!!(S.months>>MOY[a]&1)||
  !!(S.months>>MOY[b]&1));}
function tickLabel(u,e,k,step){
  var a=e[k],b=e[k+1]-1;
  if(u==='year'||u==='month')return DOM[a]===1&&MOY[a]===0&&year(a)%step===0?year(a):null;
  if(u==='week'){if(DOM[a]!==1&&MOY[a]===MOY[b])return null;var m=DOM[a]===1?a:b;
    return MON[MOY[m]]+(MOY[m]===0?' '+year(m):'');}
  return [1,8,15,22].indexOf(DOM[a])>=0?DOM[a]+' '+MON[MOY[a]]:null;
}
// With totals, the log overview adds a line for the share of each year still needing an ID (0 at
// the axis, 100% at the top), and the linear detail draws all records as faint bars behind. Years with
// fewer than SHARE_MIN records in all get no point, so a lone early record cannot draw a spike.
var SHARE_MIN=30;
function bars(cv,e,h,log,u){
  var w=cv.clientWidth,ax=13,r=devicePixelRatio||1,c=cv.getContext('2d'),n=e.length-1,sw=w/n;
  cv.width=w*r;cv.height=h*r;c.setTransform(r,0,0,r,0,0);c.clearRect(0,0,w,h);
  var v=[],tv=[],m=1,k,H=h-ax-2,bw=Math.max(sw-(sw>3?1:0),1);
  for(k=0;k<n;k++){v.push(count(e[k],e[k+1]-1));tv.push(total(e[k],e[k+1]-1));
    m=Math.max(m,v[k],log?0:tv[k]||0);}
  if(!log){c.fillStyle=tok('--total');
    for(k=0;k<n;k++)if(tv[k]){var ty=Math.max(1.5,tv[k]/m*H);
      c.globalAlpha=on(e[k],e[k+1]-1)?1:.25;c.fillRect(k*sw,h-ax-ty,bw,ty);}}
  c.fillStyle=tok('--bar');
  for(k=0;k<n;k++){
    var f=log?Math.log(1+v[k])/Math.log(1+m):v[k]/m,y=v[k]?Math.max(1.5,f*H):0;
    c.globalAlpha=on(e[k],e[k+1]-1)?1:.25;c.fillRect(k*sw,h-ax-y,bw,y);
  }
  if(log){c.globalAlpha=1;c.strokeStyle=c.fillStyle=tok('--share');c.lineWidth=1.5;c.beginPath();
    var pen=false;
    for(k=0;k<n;k++){if(!(tv[k]>=SHARE_MIN)){pen=false;continue;}
      var x=k*sw+bw/2,sy=h-ax-Math.min(1,v[k]/tv[k])*H;
      if(pen)c.lineTo(x,sy);else c.moveTo(x,sy);pen=true;}
    c.stroke();}
  c.globalAlpha=1;c.fillStyle=tok('--muted');c.font='10px system-ui,sans-serif';c.textAlign='left';
  var per=u==='year'?1:u==='month'?12:0,step=1,end=-1e9;
  if(per)step=[1,2,5,10,20,25,50].find(function(s){return s*per*sw>=32;})||100;
  for(k=0;k<n;k++){
    var t=tickLabel(u,e,k,step),x=k*sw;if(!t)continue;
    var tw=c.measureText(t).width;if(x<end+6||x+tw>w)continue;
    c.fillRect(x,h-ax,1,3);c.fillText(t,x+2,h-2);end=x+tw+2;
  }
}
function draw(){
  if(!drag)VIEW=viewOf();
  HU=unitOf(VIEW);HE=edges(HU,VIEW[0],VIEW[1]);
  bars(over,OE,34,true,'year');bars(hist,HE,64,false,HU);
  var cap='One bar per '+HU+', '+dlabel(VIEW[0])+' to '+dlabel(VIEW[1])+'.'+
    (TC&&(HU==='day'||HU==='week')?' Faint all-record bars show only with bars per month or year.':'');
  $('unit').textContent=cap;hist.setAttribute('aria-label',cap+' Drag to pick a period.');
}
function binAt(cv,e,x){var n=e.length-1;return Math.max(0,Math.min(n-1,Math.floor(x/(cv.clientWidth/n))));}
function brush(cv,get){
  var st=null;
  cv.addEventListener('pointerdown',function(ev){cv.setPointerCapture(ev.pointerId);
    st={x:ev.offsetX,moved:false,e:get().e};if(cv===hist)drag=st;});
  cv.addEventListener('pointermove',function(ev){
    var g=get(),e=st?st.e:g.e,k=binAt(cv,e,ev.offsetX);
    var nk=count(e[k],e[k+1]-1),tk=total(e[k],e[k+1]-1);
    htip.textContent=binLabel(g.u,e[k])+': '+nf.format(nk)+(tk?' of '+nf.format(tk)+' still need an ID ('+
      pct(nk,tk)+')':'');
    // Clamp by the tip's own width so it never pokes past the chart: overflow there adds a page
    // scrollbar, which shrinks the map and shifts its view.
    htip.hidden=false;htip.style.top=(cv.offsetTop-4)+'px';
    var hw=htip.offsetWidth/2;
    htip.style.left=Math.max(hw,Math.min(ev.offsetX,cv.clientWidth-hw))+'px';
    if(!st)return;
    if(Math.abs(ev.offsetX-st.x)>3)st.moved=true;
    if(st.moved){var a=binAt(cv,e,Math.min(st.x,ev.offsetX)),z=binAt(cv,e,Math.max(st.x,ev.offsetX));
      S.lo=e[a];S.hi=e[z+1]-1;render();}
  });
  cv.addEventListener('pointerup',function(ev){
    if(st&&!st.moved){var k=binAt(cv,st.e,ev.offsetX);S.lo=st.e[k];S.hi=st.e[k+1]-1;}
    st=null;drag=null;htip.hidden=true;render();});
  cv.addEventListener('pointerleave',function(){htip.hidden=true;});
  cv.addEventListener('dblclick',function(){S.lo=0;S.hi=LAST;render();});
}
brush(over,function(){return {e:OE,u:'year'};});
brush(hist,function(){return {e:HE,u:HU};});

// Controls.
var PRESETS=[
  ['Last 30 days',function(){return [clamp(LAST-29),LAST];}],
  ['Last 12 months',function(){return [clamp(LAST-364),LAST];}],
  ['This year',function(){return [clamp(dayOf(iso(LAST).slice(0,4)+'-01-01')),LAST];}],
  ['All dates',function(){return [0,LAST];}]
].map(function(p){var b=document.createElement('button');b.type='button';b.textContent=p[0];
  b.onclick=function(){var r=p[1]();S.lo=r[0];S.hi=r[1];render();};$('presets').appendChild(b);
  return {el:b,range:p[1]};});
$('from').min=$('to').min=iso(0);$('from').max=$('to').max=iso(LAST);
$('from').onchange=function(){var d=dayOf(this.value);if(d!=null){S.lo=clamp(d);
  if(S.lo>S.hi)S.hi=S.lo;render();}};
$('to').onchange=function(){var d=dayOf(this.value);if(d!=null){S.hi=clamp(d);
  if(S.hi<S.lo)S.lo=S.hi;render();}};
$('byObs').onclick=function(){S.up=false;refilter();};
$('byUp').onclick=function(){S.up=true;refilter();};
function chip(box,label,pressed,onclick,title){var b=document.createElement('button');
  b.type='button';b.textContent=label;if(title)b.title=title;
  b.setAttribute('aria-pressed',String(pressed));
  b.onclick=function(){b.setAttribute('aria-pressed',String(onclick()));refilter();};
  $(box).appendChild(b);return b;}
// Like the months: with no group picked every group shows. A click picks a group, the next leaves it
// out, the third clears it (see map_chips.js).
function allGroups(){return !CH.any(S.groups);}
function gname(g){return META.names[g]||g;}
// a group chip shows the plain name only, "Fish" for "Fish (Actinopterygii)"; its title has both
function gshort(g){return gname(g).replace(/ \([^)]*\)$/,'');}
function showGroups(){GB.forEach(function(b,i){var g=META.groups[i];CH.show(b,gshort(g),S.groups[g]||'');});}
var GB=META.groups.map(function(g){return CH.chip($('groups'),g,gshort(g),function(){return S.groups;},true,
  gname(g),refilter);});
$('gall').onclick=function(){S.groups={};showGroups();refilter();};
var MB_=MON.map(function(m,i){var b=chip('months',m,!!(S.months>>i&1),
  function(){S.months^=1<<i;return !!(S.months>>i&1);});b.setAttribute('aria-label',MONTH[i]);
  return b;});
function showMonths(){MB_.forEach(function(b,i){b.setAttribute('aria-pressed',String(!!(S.months>>i&1)));});}
$('mclear').onclick=function(){S.months=0;showMonths();refilter();};
// Introduced and threatened can be left out too. Exact location stays two-state: its opposite, hidden
// or over 1 km uncertain, is an OR iNaturalist's Identify filters cannot say.
var ONLYDEF=[['introduced','Introduced to BC','introduced','Species not native to BC'],
  ['threatened','Threatened','threatened','Species with a threatened status'],
  ['exact','Exact location','obscured','Hide records whose place is hidden or over 1 km uncertain']]
  .filter(function(o){return META.flags.indexOf(o[2])>=0;}),
  ONLY=ONLYDEF.map(function(o){return CH.chip($('flags'),o[0],o[1],function(){return S.only;},o[0]!=='exact',
    o[3],refilter);});
function showOnly(){ONLY.forEach(function(b,j){CH.show(b,ONLYDEF[j][1],S.only[ONLYDEF[j][0]]||'');});}
if(!ONLY.length)$('flagbox').hidden=true;
// Picked taxa: chips in the selection bar, one set with the groups. A shortcut picks, then leaves out, a set
// of taxa, and shows only once the taxa file carries the whole tree.
var PK=TX.picker({input:$('q'),list:$('sugg'),msg:$('qn'),picks:S.picks,ranks:META.ranks,
  nf:nf,taxa:function(){return TAXA;},counts:function(){return TX.counts(TAXA,P.tax,P.n);},
  onChange:function(){search();showClades();refilter();}});
var PRE=META.presets||[],CB=PRE.map(function(c){return chip('clades',c[0],false,function(){
  S.split=S.split.filter(function(n){return n!==c[0];});PK.cycle(c[1]);return PK.state(c[1])!=='';},c[2]);});
$('cladebox').hidden=!CB.length;
function showClades(){CB.forEach(function(b,k){CH.show(b,PRE[k][0],PK.state(PRE[k][1]));});}
showClades();
// An old link's q, names to keep, becomes picks once the taxa load. It kept those names within the
// groups; when every name lies inside the groups the picks alone say the same, so the groups clear.
function oldQuery(){if(!S.oldq)return;
  var got=TX.resolve(TAXA,S.oldq,TX.counts(TAXA,P.tax,P.n)),gid=META.group_ids||{},
    gs=CH.picked(S.groups,META.groups,'inc').filter(function(g){return gid[g];}).map(function(g){return gid[g];});
  if(!got.length)$('q').value=S.oldq;S.oldq='';
  got.forEach(function(p){S.picks.push(p);});
  if(got.length&&!allGroups()&&got.every(function(p){return p.not||TX.within(TAXA,p.id,gs);}))
    S.groups={},showGroups();
  showClades();}
function search(){TF=TX.filter(TAXA,S.picks);}
// The selection bar lists every chip; its Clear all clears them, and reset also the dates.
function redraw(){showGroups();showMonths();showOnly();refilter();}
function clearAll(){S.months=0;S.groups={};S.only={};S.picks.length=0;S.split=[];$('q').value='';showClades();
  search();AREA.clear();PJ.reset();redraw();}
$('reset').onclick=function(){S.lo=0;S.hi=LAST;S.up=false;clearAll();};
var BAR=CH.bar({box:$('selbar'),clear:clearAll,fallback:$('q')});
function items(){return CH.items({S:S,meta:META,terms:TX.terms({T:TAXA,picks:S.picks,groups:[],meta:META,n:0,split:S.split}),
  picker:PK,pj:PJ,months:MONTH,only:ONLYDEF,two:['exact'],area:AREA,redraw:redraw});}
// Each folding section's summary counts its active chips, so a folded one still says what it holds.
function tally(){var n=function(st){var c=0;for(var k in st)if(st[k])c++;return c;},m=0,i,e;
  for(i=0;i<12;i++)m+=S.months>>i&1;
  [['ntaxa',S.picks.length],['nonly',n(S.only)],['nproj',PJ.items().length],['ngroups',n(S.groups)],
    ['nmonths',m]].forEach(function(d){e=$(d[0]);if(e)e.textContent=d[1]?String(d[1]):'';});}
// Taxa and Show only start open, Groups, Projects and Months folded; a section opens as the viewer
// last left it, and one a link filters on opens regardless (projects only know theirs once loaded).
var FOLD='wtid-map-fold';
(function(){var saved={};try{saved=JSON.parse(localStorage.getItem(FOLD))||{};}catch(e){}
  var on={grpsec:CH.any(S.groups),monsec:!!S.months,flagbox:CH.any(S.only),projbox:/(^|[#&])projects=/.test(location.hash)};
  [].forEach.call(document.querySelectorAll('details.sec'),function(d){
    if(on[d.id])d.open=true;else if(d.id in saved)d.open=saved[d.id];
    d.addEventListener('toggle',function(){saved[d.id]=d.open;
      try{localStorage.setItem(FOLD,JSON.stringify(saved));}catch(e){}});});})();
$('toggle').onclick=function(){var o=$('side').classList.toggle('open');
  this.setAttribute('aria-expanded',String(o));};

// Hover and click.
function names(i){var t=TAXA&&P.tax[i]!==NOTAX?TAXA[P.tax[i]]:null,g=META.groups[P.grp[i]];
  return {common:t&&t[1]||'',latin:t&&t[0]||META.names[g]||g,group:META.names[g]||g,
    rank:t&&t[3]!==255?META.ranks[t[3]]:''};}
function badges(i){var f=P.fl[i],out=[];
  if(f&FLAG.introduced)out.push(['Introduced','']);if(f&FLAG.threatened)out.push(['Threatened','']);
  if(f&FLAG.obscured)out.push(['Location hidden','warn']);
  else if(f&FLAG.imprecise)out.push(['Location over 1 km uncertain','warn']);
  return out.map(function(b){return '<span class="badge '+b[1]+'">'+b[0]+'</span>';}).join('');}
function idText(i){var n=P.ids[i]&15,a=P.ids[i]>>4;
  return n?(n>=15?'15+':n)+(n===1?' ID':' IDs')+(a?', '+(a>=15?'15+':a)+' agreeing':''):'No IDs yet';}
function when(v,early){return v===NO?'unknown':early?'before 1900':iso(v);}
// The hover tip sits below right of the pointer and flips to the other side of it near the window's
// right or bottom edge. It is fixed to the window above the record card, so neither the map's
// edges, the time bar nor an open card hide it.
var ptip=document.createElement('div');ptip.className='ptip';ptip.hidden=true;
function tip(o){
  if(!P.n||o.index<0||!o.layer||o.layer.id!=='pts'||AREA.busy()){ptip.hidden=true;return;}
  var i=o.index,t=names(i);
  ptip.innerHTML='<b>'+esc(t.common||t.latin)+'</b>'+(t.common?'<br><i>'+esc(t.latin)+'</i>':'')+
    '<br>'+esc(t.group)+' · observed '+when(P.obs[i],P.fl[i]&FLAG.earlyObs)+'<br>'+idText(i)+'<div class="badges">'+
    badges(i)+'</div>';
  ptip.hidden=false;
  var r=map.getContainer().getBoundingClientRect(),px=r.left+o.x,py=r.top+o.y,
    w=ptip.offsetWidth,h=ptip.offsetHeight,g=12,
    x=px+g+w>innerWidth-4?px-g-w:px+g,y=py+g+h>innerHeight-4?py-g-h:py+g;
  ptip.style.left=Math.max(4,x)+'px';ptip.style.top=Math.max(4,y)+'px';
}
var ctl=null,card=$('card');
function pick(o){if(!P.n||o.index<0||o.layer.id!=='pts'||AREA.busy())return;openCard(o.index);}
// A record deleted from iNaturalist leaves the map once its card closes, so the card can say why.
// The card opens beside its point, on the side with room, and follows the point as the map moves.
// It is fixed to the window, so the time bar and the map's edges never clip it. On a phone it is a
// bottom sheet instead.
var NARROW=matchMedia('(max-width:760px)');
function place(){
  if(SEL<0||card.hidden)return;var s=card.style;
  if(NARROW.matches){s.left=s.top=s.right='';return;}
  var r=map.getContainer().getBoundingClientRect(),p=map.project([P.pos[2*SEL],P.pos[2*SEL+1]]),
    x=r.left+p.x,y=r.top+p.y,w=card.offsetWidth,h=card.offsetHeight,g=14,m=8,
    left=x+g+w<=innerWidth-m?x+g:x-g-w>=m?x-g-w:innerWidth-m-w;
  s.right='auto';s.left=Math.max(m,left)+'px';
  s.top=Math.max(m,Math.min(y-h/2,innerHeight-m-h))+'px';
}
function closeCard(){SEL=-1;card.hidden=true;if(ctl)ctl.abort();
  if(GONE_NEW){GONE_NEW=false;refilter();}else{layers();writeHash();}}
function openCard(i){
  S.rec=null;$('record-note').hidden=true;SEL=i;var t=names(i),id=P.id[i],lag=P.obs[i]!==NO&&P.up[i]!==NO&&!(P.fl[i]&(FLAG.earlyObs|FLAG.earlyUp))?P.up[i]-P.obs[i]:null;
  card.innerHTML='<button type="button" class="x" aria-label="Close">×</button>'+
    '<div class="ph" id="ph">Loading photo</div><div class="body">'+
    '<h2>'+esc(t.common||t.latin)+'</h2><p class="latin"><i>'+esc(t.latin)+'</i>'+
    (t.rank?' ('+esc(t.rank)+')':'')+'</p><div class="badges">'+badges(i)+'</div><dl>'+
    '<dt>Group</dt><dd>'+esc(t.group)+'</dd><dt>Observed</dt><dd>'+when(P.obs[i],P.fl[i]&FLAG.earlyObs)+'</dd>'+
    '<dt>Uploaded</dt><dd>'+when(P.up[i],P.fl[i]&FLAG.earlyUp)+(lag>0?' ('+nf.format(lag)+(lag===1?' day':' days')+
    ' later)':'')+'</dd><dt>IDs</dt><dd>'+idText(i)+'</dd><dt>Place</dt><dd id="place">…</dd></dl>'+
    '<div id="now"></div><a class="btn" target="_blank" rel="noopener" '+
    'href="https://www.inaturalist.org/observations/'+id+'">Help identify on iNaturalist</a>'+
    '<p class="attr" id="attr"></p></div>';
  if(META.atlas_url){var sample=document.createElement('a');sample.className='btn';sample.textContent='Explore image sample';sample.href=MapLinks.atlasHref(META.atlas_url,location.href,META.groups[P.grp[i]],id);sample.target='_blank';sample.rel='noopener noreferrer';var note=document.createElement('p');note.className='attr';note.textContent='Opens a separate frozen image sample, without changing this map or its record list.';card.querySelector('.body').appendChild(sample);card.querySelector('.body').appendChild(note);}
  card.hidden=false;card.querySelector('.x').onclick=closeCard;card.querySelector('.x').focus();
  place();
  layers();writeHash();
  if(ctl)ctl.abort();ctl=new AbortController();
  fetch('https://api.inaturalist.org/v1/observations/'+id,{signal:ctl.signal})
    .then(function(r){if(r.status===404)return {results:[]};
      if(!r.ok)throw new Error('HTTP '+r.status);return r.json();})
    .then(function(j){
      var o=j.results&&j.results[0];
      if(!o){$('ph').textContent='This record is no longer on iNaturalist, so it leaves the map';
        $('place').textContent='';GONE[id]=1;GONE_NEW=true;return;}
      var ph=o.photos&&o.photos[0];
      if(ph){var img=new Image();img.alt='Photo of '+(t.common||t.latin);
        img.onload=place;img.src=String(ph.url).replace('/square.','/medium.');$('ph').textContent='';
        $('ph').appendChild(img);$('attr').textContent=ph.attribution||'';}
      else $('ph').textContent='No photo on iNaturalist now';
      $('place').textContent=o.place_guess||'Not given';
      var now=[],ot=o.taxon||{};
      if(o.quality_grade&&o.quality_grade!=='needs_id')now.push('Now '+
        (o.quality_grade==='research'?'Research Grade':'casual')+', so it no longer needs an ID.');
      if(ot.name&&ot.name!==t.latin)now.push('Now identified as '+
        (ot.preferred_common_name?ot.preferred_common_name+' (':'')+ot.name+
        (ot.preferred_common_name?')':'')+'.');
      if(now.length){var p=document.createElement('p');p.className='now';p.textContent=
        'Since this map was built: '+now.join(' ');$('now').appendChild(p);}
      place();
    }).catch(function(e){if(e.name==='AbortError')return;
      $('ph').textContent='Could not load the photo';$('place').textContent='Unknown';});
}
addEventListener('keydown',function(e){if(e.key==='Escape'&&!card.hidden)closeCard();});

// Map, then data: names and the recent shard first, older shards after.
var b=META.bbox,opts={container:'map',style:dark?BASEMAPS.dark:BASEMAPS.light,
  attributionControl:{compact:true}};
if(S.at){opts.center=[S.at[2],S.at[1]];opts.zoom=S.at[0];}
else{opts.bounds=[[b[0],b[1]],[b[2],b[3]]];opts.fitBoundsOptions={padding:20};}
var map=new maplibregl.Map(opts);document.body.appendChild(ptip);
map.addControl(new maplibregl.NavigationControl({showCompass:false}));
var overlay=new deck.MapboxOverlay({interleaved:false,pickingRadius:8,layers:[]});map.addControl(overlay);
var AREA=MapArea({map:map,S:S,LAST:LAST,hash:S.area,park:S.park,bcpark:S.bcpark,fit:!S.at,
  place:META.place_id,parks:META.parks,get:get,P:function(){return P;},keep:function(){return KEEP;},
  change:refilter,sel:function(){return sel().join('&');},
  step:IdStep({a:$('identify'),prev:$('idprev'),next:$('idnext'),build:iso(LAST)})});
// Zoomed out, blending millions of overlapping dots takes about 250 ms a frame, so while the map
// moves it draws one record in ten, spread across BC by the id order, and all of them once it stops.
var THIN=false,THIN_BELOW=8;
map.on('move',function(){if(!THIN&&map.getZoom()<THIN_BELOW){THIN=true;layers();}});
map.on('move',place);
map.on('moveend',function(){if(THIN){THIN=false;layers();}link();writeHash();});
map.getContainer().addEventListener('pointerleave',function(){ptip.hidden=true;});
addEventListener('resize',function(){draw();place();});
var HINT=$('hint').textContent,HINT_TOT='Top: records still needing an ID per year, on a log scale; the line is the '+
  'share of all BC records with photos that still need one. Click or drag across years to zoom in. Bottom: '+
  'the period in view, faint bars all records, solid bars those still needing an ID. Drag to pick a period, '+
  'click a bar to zoom into it, double-click for all dates.';
render();
PJ.load(get);
var names0=get(META.taxa).then(function(buf){TAXA=JSON.parse(new TextDecoder().decode(buf));
  oldQuery();search();showClades();refilter();});
// One shard downloads at a time, so on a slow link the recent shard gets all the bandwidth.
var chain=Promise.resolve(),prev=Promise.resolve();
META.shards.forEach(function(s,k){
  var got=prev.then(function(){return get(s.file);});prev=got;
  chain=chain.then(function(){return got;}).then(function(buf){
    addShard(buf,s.n);
    $('more').textContent=k<META.shards.length-1?'Loading older records…':'';
    if(k===0)return names0;
  }).then(function(){
    refilter();
    if(S.rec&&SEL<0){var requested=S.rec,resolution=MapLinks.resolveRecord(P.id,requested,k===META.shards.length-1);
      if(resolution.index>=0){var i=resolution.index;openCard(i);
        if(S.fly){S.fly=false;map.flyTo({center:[P.pos[2*i],P.pos[2*i+1]],zoom:12});}}
      else if(resolution.missing){var notice=$('record-note');notice.hidden=false;notice.textContent='Requested observation '+requested+' is not in this map snapshot. No replacement has been selected. ';var original=document.createElement('a');original.href='https://www.inaturalist.org/observations/'+requested;original.textContent='View the requested observation on iNaturalist';original.target='_blank';original.rel='noopener noreferrer';notice.appendChild(original);writeHash();}}

  });
});
chain.catch(function(e){$('count').textContent='Could not load the records ('+e.message+')';});
})();
