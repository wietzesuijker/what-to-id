/* Navigation between public exploratory pages, independent of any record queue. */
var MapLinks=(function(){
'use strict';
function atlasHref(target,base,group,id){var url=new URL(target,base);
  url.hash=new URLSearchParams({group:group,id:String(id)}).toString();return url.href;}
function resolveRecord(ids,requested,complete){
  for(var i=0;i<ids.length;i++)if(ids[i]===requested)return {index:i,missing:false};
  return {index:-1,missing:!!complete};}
return {atlasHref:atlasHref,resolveRecord:resolveRecord};
})();
if(typeof module!=='undefined')module.exports=MapLinks;
