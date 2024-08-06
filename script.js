$(document).ready(function() {
    const width = 800;
    const height = 600;

    const svg = d3.select("#map")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    let locations = [
        {id: 0, x: 100, y: 100, type: "depot"}
    ];

    function updateMap() {
        const circles = svg.selectAll("circle")
            .data(locations, d => d.id);

        circles.enter()
            .append("circle")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", 10)
            .attr("fill", d => d.type === "depot" ? "red" : "blue");

        circles.exit().remove();
    }

    function addRandomLocation() {
        const newLocation = {
            id: locations.length,
            x: Math.random() * (width - 20) + 10,
            y: Math.random() * (height - 20) + 10,
            type: "delivery"
        };
        locations.push(newLocation);
        updateMap();
    }

    $("#add-location").click(addRandomLocation);

    $("#optimize").click(function() {
        $.ajax({
            url: "/optimize",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({locations: locations}),
            success: function(response) {
                drawRoute(response.route);
                updateStats(response.stats);
            }
        });
    });

    $("#reset").click(function() {
        locations = [locations[0]];  // Keep only the depot
        updateMap();
        svg.selectAll("line").remove();
        $("#stats").empty();
    });

    function drawRoute(route) {
        svg.selectAll("line").remove();

        for (let i = 0; i < route.length - 1; i++) {
            const start = locations[route[i]];
            const end = locations[route[i+1]];

            svg.append("line")
                .attr("x1", start.x)
                .attr("y1", start.y)
                .attr("x2", end.x)
                .attr("y2", end.y)
                .attr("stroke", "green")
                .attr("stroke-width", 2);
        }
    }

    function updateStats(stats) {
        $("#stats").html(`
            <h3>Route Statistics</h3>
            <p>Total Distance: ${stats.total_distance.toFixed(2)}</p>
            <p>Number of Locations: ${stats.num_locations}</p>
        `);
    }

    updateMap();
});