import sys, urllib.parse, urllib.request, xml.etree.ElementTree as ET, time, json
NS={'a':'http://www.w3.org/2005/Atom'}
def search(q, n=8):
    url="http://export.arxiv.org/api/query?"+urllib.parse.urlencode(
        {"search_query":q,"start":0,"max_results":n,"sortBy":"relevance"})
    for _ in range(3):
        try:
            raw=urllib.request.urlopen(url,timeout=45).read()
            break
        except Exception as e:
            time.sleep(3); raw=None
    if not raw: return []
    root=ET.fromstring(raw); out=[]
    for e in root.findall('a:entry',NS):
        out.append({
          "id": e.find('a:id',NS).text.split('/abs/')[-1],
          "title":" ".join(e.find('a:title',NS).text.split()),
          "date": e.find('a:published',NS).text[:10],
          "summary":" ".join(e.find('a:summary',NS).text.split())[:400]})
    return out
queries=json.load(open(sys.argv[1]))
for q in queries:
    print("\n\n######## QUERY:", q)
    for r in search(q):
        print(f"- [{r['date']}] {r['id']} :: {r['title']}")
        print(f"    {r['summary'][:300]}")
    time.sleep(3)
