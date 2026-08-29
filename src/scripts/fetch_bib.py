"""Pull authors/venue for arXiv IDs so bibliography entries carry real metadata."""
import sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET, time, json, re
NS={'a':'http://www.w3.org/2005/Atom','arxiv':'http://arxiv.org/schemas/atom'}
ids=sys.argv[1:]
out={}
for i in range(0,len(ids),20):
    url="http://export.arxiv.org/api/query?"+urllib.parse.urlencode(
        {"id_list":",".join(ids[i:i+20]),"max_results":50})
    raw=urllib.request.urlopen(url,timeout=60).read()
    for e in ET.fromstring(raw).findall('a:entry',NS):
        aid=e.find('a:id',NS).text.split('/abs/')[-1]
        base=aid.split('v')[0]
        auth=[a.find('a:name',NS).text for a in e.findall('a:author',NS)]
        jr=e.find('arxiv:journal_ref',NS)
        out[base]=dict(id=base, title=" ".join(e.find('a:title',NS).text.split()),
                       authors=auth, year=e.find('a:published',NS).text[:4],
                       journal_ref=jr.text if jr is not None else None)
    time.sleep(3)
print(json.dumps(out,ensure_ascii=False,indent=1))
