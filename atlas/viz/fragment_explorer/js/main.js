
// Chart options
var r_small = 4;
var r_select = 8;
var select_color = "#fcd303";

var width = 600;
var height = 600;


function filter_data(data, sx, sy, t) {
    var scope = [];

    for (var i = 0; i < data.length; ++i) {
        var x = data[i].x;
        var y = data[i].y;

        var tx = t.applyX(sx(x));
        var ty = t.applyY(sy(y));

        if (tx >= 0 && tx < 600 && ty >= 0 && ty < 600) {
            scope.push(data[i]);
        }
    }

    return scope;
}

// Outdated SVG version (much slower)
function draw_chart_svg(data) {
    // Compute bounds.
    let ax = d3.min(data, d => d.x) - 1;
    let bx = d3.max(data, d => d.x) + 1;

    let ay = d3.min(data, d => d.y) - 1;
    let by = d3.max(data, d => d.y) + 1;

    let az = d3.min(data, d => d.z) - 1;
    let bz = d3.max(data, d => d.z) + 1;

    var x = d3.scaleLinear()
        .domain([ax, bx])
        .range([ 0, width ]);

    var y = d3.scaleLinear()
        .domain([ay, by])
        .range([ height, 0]);

    var colors = d3.scaleLinear()
        .domain([az, bz])
        .range(["#e67e22", "#3498db"])
        .interpolate(d3.interpolateHcl);

    function mouseover(d) {
        var dat = d3.select(this)
            .data()[0];

        d3.select(this)
            .attr("r", r_select)
            .style("fill", select_color)

        d3.select('#mol_viewer')
            .transition()
            .style('opacity', 1)

        var x = d.clientX;
        var y = d.clientY;
    
        d3.select('#mol_viewer')
            .style('left', x + 'px')
            .style('top', y + 'px')

        draw_mol(dat.smiles, dat.idx);
    }

    function mouseleave(d) {
        d3.select(this)
            .attr("r", r_small)
            .style("fill", d => colors(d.z))

        d3.select('#mol_viewer')
            .transition()
            .style('opacity', 0)
    }

    var svg = d3.select("#tsne_plot")
        .append("svg")
        .attr("viewBox", "0 0 600 600")

    var g = svg.append('g')

    var circles = g.selectAll("dot")
        .data(data)
        .enter()
        .append("circle")

    circles
        .attr("cx", function (d) { return x(d.x); } )
        .attr("cy", function (d) { return y(d.y); } )
        .attr("r", r_small)
        .style("fill", d => colors(d.z))
        .on("mouseover", mouseover )
        .on("mouseleave", mouseleave )

    svg.call(d3.zoom().on("zoom", function (event) {
        circles
            .attr("cx", d => event.transform.applyX(x(d.x)))
            .attr("cy", d => event.transform.applyY(y(d.y)))            
    }))

    window.addEventListener("mousemove", mousemove )
}

function draw_chart_webgl(data) {
    let ax = d3.min(data, d => d.x) - 1;
    let bx = d3.max(data, d => d.x) + 1;

    let ay = d3.min(data, d => d.y) - 1;
    let by = d3.max(data, d => d.y) + 1;

    let az = d3.min(data, d => d.z);
    let bz = d3.max(data, d => d.z);

    var xScale = d3.scaleLinear()
        .domain([ax, bx])
        .range([ 0, width ]);

    var yScale = d3.scaleLinear()
        .domain([ay, by])
        .range([ height, 0]);

    var colors = d3.scaleLinear()
        .domain([az, (az + bz)/2, bz])
        .range(["#1a2a6c", "#b21f1f", "#fdbb2d"])
        .interpolate(d3.interpolateHcl);

    const pointSeries = fc
        .seriesWebglPoint()
        .equals((previousData, currentData) => previousData === currentData)
        .size(20)
        .crossValue(d => d.x)
        .mainValue(d => d.y);

    const webglColor = (color, alpha) => {
        const { r, g, b, opacity } = d3.color(color).rgb();
        return [r / 255, g / 255, b / 255, alpha];
    };

    const fillColor = fc
        .webglFillColor()
        .value(d => webglColor(colors(d.z), 0.8))
        .data(data);
      
    pointSeries.decorate(program => fillColor(program));

    const xScaleOriginal = xScale.copy();
    const yScaleOriginal = yScale.copy();
    
    const zoom = d3
        .zoom()
        .on("zoom", (event) => {
            xScale.domain(event.transform.rescaleX(xScaleOriginal).domain());
            yScale.domain(event.transform.rescaleY(yScaleOriginal).domain());
            redraw();
        });

    quadtree = d3.quadtree()
        .x(d => d.x)
        .y(d => d.y)
        .addAll(data);

    function mousemove(event) {
        const x = xScale.invert(event.clientX);
        const y = yScale.invert(event.clientY);

        const point = quadtree.find(x, y);

        const px = xScale(point.x);
        const py = yScale(point.y);

        // console.log(px, py);

        const dx = (event.clientX-px);
        const dy = (event.clientY-py);
        const dist = (dx*dx) + (dy*dy);
        // console.log(dist);

        if (dist > 2000) {
            d3.select('#mol_viewer')
                // .transition()
                .style('opacity', 0)

            d3.select('#zinc_label')
                .html('')
        } else {
            d3.select('#mol_viewer')
                // .transition()
                .style('opacity', 1)

            d3.select('#mol_viewer')
                .style('left', event.clientX + 'px')
                .style('top', event.clientY + 'px')

            d3.select('#zinc_label')
                .html('<b>' + point.zinc_id + '</b> ' + point.smiles)

            draw_mol(point.smiles, point.idx);
        }
    }
    
    const chart = fc
        .chartCartesian(xScale, yScale)
        .webglPlotArea(pointSeries)
        .decorate(sel => {
            sel.enter()
                .selectAll('.x-axis')
                .on('measure.range', event => {
                    xScaleOriginal.range([0, event.detail.width]);
                });
            sel.enter()
                .selectAll('.y-axis')
                .on('measure.range', event => {
                    yScaleOriginal.range([event.detail.height, 0]);
                });
            sel.enter()
                .selectAll('.plot-area')
                .call(zoom)
                .on('mousemove', mousemove)
        })
    
    const redraw = () => {
        d3.select("#tsne_plot")
            .datum(data)
            .call(chart);
    };

    redraw();
}

d3.csv("/maps/tsne_gcn.csv", d3.autoType).then(draw_chart_webgl);



let viewer = $3Dmol.createViewer(
    document.getElementById('mol_viewer'), 
    {}
);
viewer.setBackgroundColor(0,0);



var OpenBabel = OpenBabelModule();
var ob_ready = false;
OpenBabel.onRuntimeInitialized = function() {
    ob_ready = true;
    console.log('Open babel loaded');
}


var curr_smiles = '';
var curr_idx = -1;

function draw_mol(smiles, index) {
    if (!ob_ready) return;

    if (smiles == curr_smiles && index == curr_idx) return;

    curr_smiles = smiles;
    curr_idx = index;

    // Generate 2d sdf.
    var conv = new OpenBabel.ObConversionWrapper()
    var mol = new OpenBabel.OBMol();

    conv.setInFormat('', 'smi');
    conv.readString(mol, smiles);

    var gen2d = OpenBabel.OBOp.FindType('Gen2D');
    gen2d.Do(mol, '');
    conv.setOutFormat('', 'sdf');

    var sdf = conv.writeString(mol, true);

    // Fetch atom coordinates.
    var atom = mol.GetAtom(index + 1);
    var ax = atom.GetX();
    var ay = atom.GetY();
    var az = atom.GetZ();

    viewer.clear();
    viewer.addModel(sdf, 'sdf');
    viewer.setStyle({'model': 0}, {'stick':{}})

    // Highlight selected atom.
    viewer.addSphere({
        'center': {'x': ax, 'y': ay, 'z': az},
        'radius': 0.5,
        'color': 'green',
        'opacity': 0.8
    })
    viewer.zoomTo({'serial': index})
    viewer.zoom(0.5);
    viewer.render();
}
