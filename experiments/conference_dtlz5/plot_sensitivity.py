from pathlib import Path
import csv
from itertools import groupby
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

work=Path(__file__).resolve().parent
folder=work
with (work/"dtlz5_5_20_figure4.csv").open(newline="") as source:
    rows=list(csv.DictReader(source))
scores={(float(r["step_size"]), int(r["points_per_initial_point"])):int(r["accuracy_percent"]) for r in rows}
steps=sorted({s for s,n in scores},reverse=True)
counts=sorted({n for s,n in scores})
assert steps==[.05,.03,.02,.01,.008,.005,.003,.002,.001]
assert counts==[5,10,20,30,40,50,80,100,150] and len(scores)==81
assert all(0<=a<=100 and a%10==0 for a in scores.values())

# Use the original Arial font when available; Helvetica provides a PDF-native fallback.
import os
font_path = Path(os.environ.get("ORCA_FIGURE_FONT", "C:/Windows/Fonts/arial.ttf"))
if font_path.is_file():
    pdfmetrics.registerFont(TTFont("FigureSans", str(font_path)))
else:
    pdfmetrics.registerFont(pdfmetrics.Font("FigureSans", "Helvetica", "WinAnsiEncoding"))
stops=[(0,"#fff5d9"),(20,"#dce7b8"),(40,"#a2d2b6"),(60,"#5cb3ad"),(80,"#2a7d88"),(100,"#1b425d")]
colorscale=[colors.HexColor(h) for _,h in stops]
def swatch(v):
    for ix,((lo,_),(hi,_)) in enumerate(zip(stops,stops[1:])):
        if v<=hi:
            amount=(v-lo)/(hi-lo)
            return colors.Color(*[a+amount*(b-a) for a,b in zip(colorscale[ix].rgb(),colorscale[ix+1].rgb())])
    raise ValueError(v)
p=canvas.Canvas(str(folder/"fig4_sensitivity.pdf"),pagesize=(380,270),pageCompression=1,invariant=True)
p.setTitle("Accuracy of objective grouping for DTLZ5(5,20)")
p.setAuthor("Hongxuan Wang and Andrew Allman")
p.setSubject("Historical data for Figure 3 of Wang and Allman (2025), DOI 10.69997/sct.108915, showing 5 to 150 added points per seed")
ink=colors.HexColor("#293436")
def label(x,y,s,size=9,align="center"):
    p.setFillColor(ink)
    p.setFont("FigureSans",size)
    {"center":p.drawCentredString,"left":p.drawString,"right":p.drawRightString}[align](x,y,str(s))

x0,y0,x1,y1=42,48,316,259
dx,dy=(x1-x0)/len(steps),(y1-y0)/len(counts)
# Paint adjacent vector fills with overlapping underlays to avoid PDF-viewer hairlines.
p.setFillColor(swatch(0));p.rect(x0,y0,x1-x0,y1-y0,fill=1,stroke=0)
for j,n in enumerate(counts):
    values=[scores[(s,n)] for s in steps]
    i=0
    for value,cells in groupby(values):
        length=sum(1 for _ in cells)
        right=x0+(i+length)*dx
        top=y0+(j+1)*dy
        if i+length<len(steps): right+=0.65
        if j+1<len(counts): top+=0.65
        left=x0+i*dx
        bottom=y0+j*dy
        p.setFillColor(swatch(value))
        p.rect(left,bottom,right-left,top-bottom,fill=1,stroke=0)
        i+=length
axis=colors.HexColor("#536061")
p.setLineWidth(.42);p.setStrokeColor(axis)
p.rect(x0,y0,x1-x0,y1-y0,fill=0,stroke=1)
for j,n in enumerate(counts):
    y=y0+(j+.5)*dy
    p.line(x0-2.3,y,x0,y)
    label(x0-5,y-3,n,9.2,"right")
for i,step in enumerate(steps):
    x=x0+(i+.5)*dx
    p.line(x,y0-2.3,x,y0)
    p.saveState();p.translate(x+1.7,y0-6.5);p.rotate(45)
    label(0,0,f"{step:g}",8.7,"right")
    p.restoreState()
label((x0+x1)/2,7.1,"Step Size",11.2)
p.saveState();p.translate(10.5,(y0+y1)/2);p.rotate(90)
label(0,0,"Number of Selected Points",10.3)
p.restoreState()
bx,bw=328,8
p.saveState()
path=p.beginPath();path.rect(bx,y0,bw,y1-y0);p.clipPath(path,stroke=0)
p.linearGradient(bx,y0,bx,y1,colorscale,positions=[s/100 for s,_ in stops],extend=False)
p.restoreState()
p.setLineWidth(.34);p.setStrokeColor(axis);p.rect(bx,y0,bw,y1-y0,stroke=1,fill=0)
for pct in range(0,101,20):
    y=y0+(y1-y0)*pct/100
    p.line(bx+bw,y,bx+bw+2.3,y);label(bx+bw+4.6,y-3,pct,8.6,"left")
p.saveState();p.translate(371.7,(y0+y1)/2);p.rotate(90);label(0,0,"Accuracy (%)",10.6);p.restoreState()
p.showPage();p.save()

