
function populateSwarmList(swarm_list, dict) {
    swarm_names = dict['swarm']
    for (var i = 0; i < swarm_names.length; i++) {
        // Create the list item:
        var text = document.createTextNode(swarm_names[i])

        var item = document.createElement('a');
        item.className = 'nav-link';
        item.setAttribute('data-toggle', 'pill');
        item.href = '#v-pills-' + swarm_names[i];
        item.setAttribute('aria-controls', 'v-pills-' + swarm_names[i]);
        item.setAttribute('role', 'tab')
        item.id = 'v-pills-' + swarm_names[i] + '-tab'

        item.appendChild(text)

        var spinner = document.createElement('span');
        spinner.className = 'spinner-grow text-secondary spinner-grow-sm ml-2';
        spinner.style.display = 'none';
        spinner.id = 'spinner-' + swarm_names[i];
        item.appendChild(spinner);

        // Add it to the list:
        swarm_list.appendChild(item);
    }
}

function getSwarmList() {
    deff = $.ajax({
        type: "GET", 
        url: "http://3.83.149.203:5000/swarm",
        async: true, 
        datatype: "json",
        success: function(result, status, xhr){
            return result;
        },
        error: function(xhr, status, err) {
          console.log(status + ": " + err);
        }
      });
    return deff
}

function getSwarmInfo(name) {
    deff = $.ajax({
        type: "GET", 
        url: "http://3.83.149.203:5000/devices",
        async: true, 
        data: {
            'swarm-name': name
        },
        datatype: "json",
        success: function(result, status, xhr){
            return result;
        },
        error: function(xhr, status, err) {
          console.log(status + ": " + err);
        }
      });
    return deff
}

function getSwarmMetric(name) {
    deff = $.ajax({
        type: "GET", 
        url: "http://3.83.149.203:5000/swarm-metric",
        async: true, 
        data: {
            'swarm-name': name
        },
        datatype: "json",
        success: function(result, status, xhr){
            return result;
        },
        error: function(xhr, status, err) {
          console.log(status + ": " + err);
        }
      });
    return deff
}

function getRunningSwarms() {
    deff = $.ajax({
        type: "GET", 
        url: "http://3.83.149.203:5000/check-running-swarm",
        async: true, 
        datatype: "json",
        success: function(result, status, xhr){
            return result;
        },
        error: function(xhr, status, err) {
          console.log(status + ": " + err);
        }
      });
    return deff
}

function appendSwarmToTable(table, swarm_data) {
    while (table.firstChild) {
        table.removeChild(table.firstChild);
    }
    for (var i=0; i < swarm_data.length; i++) {
        appendDeviceToTable(table, swarm_data[i])
    }
}

function appendDeviceToTable(table, data) {
    var row = document.createElement('tr');
    row.setAttribute('class', 'device-row')
    var row_html = '<th scope="row">';
    row_html += data['DeviceId']['N'];
    row_html += '</th>';
    var key_type = [['DeviceStatus', 'S'], ['EncIdx', 'N'], ['Hostname', 'S']];
    for (var i = 0; i < key_type.length; i++) {
        row_html += '<td>';
        if (key_type[i][0] == 'Hostname') {
            row_html += data[key_type[i][0]][key_type[i][1]].split('-')[2];
            row_html += '-' + data[key_type[i][0]][key_type[i][1]].split('-')[3];
        }
        else {
            row_html += data[key_type[i][0]][key_type[i][1]];
        }
        row_html += '</td>';
    }
    row.innerHTML = row_html;
    if (data['DeviceStatus']['S'] == 'Running') {
        row.style.backgroundColor = '#409FFF4D';
        row.style.color = '#F3F4F5'
    }
    else if (data['DeviceStatus']['S'] == 'Error') {
        row.style.backgroundColor = '#D95757';
        row.style.color = '#F3F4F5'
    }
    else {
        row.style.backgroundColor = '#1A1F29';
        row.style.color = '#CCCAC2'
    }
    row.onclick = function() {
        console.log(data['DeviceId']['N'] + ' clicked')
    };
    row.appendChild(getProgressbarSpace(data['DeviceStatus']['S'], data['EncIdx']['N'], data['TotalEncIdx']['N']))
    table.appendChild(row)
}

function getProgressbarSpace(status, encIdx, totalEncIdx) {
    var percentage = encIdx / totalEncIdx * 100;
    if (status == 'Finished') {
        percentage = 100;
    }
    console.log(percentage)
    var progressbar_space = document.createElement('td');
    progressbar_space.setAttribute('style', 'vertical-align: middle;')
    var progress = document.createElement('span');
    progress.className = 'progress';
    progress.setAttribute('style', 'height: 5px; width: 85%;')
    var progressbar = document.createElement('span');
    progressbar.className = 'progress-bar progress-bar-striped bg-success';
    if (percentage < 100) {
        progressbar.className += ' progress-bar-animated';
    }
    progressbar.setAttribute('style', 'width:'+ percentage + '%;')
    progressbar.setAttribute('role', 'progressbar')
    progress.appendChild(progressbar)
    progressbar_space.appendChild(progress)
    return progressbar_space
}

window.addEventListener('DOMContentLoaded', (event) => {
    var els = document.querySelectorAll('.device-row');
    els.forEach(function(cell) {
      console.log(cell);
    })
  })

$(document).ready(function () {
    console.log('Hello, swarm!')

    var selected_swarm_name = null;

    // populate swarm name list in sidebar
    getSwarmList().done(function(data) {
            console.log('got response' + data)
            populateSwarmList(document.getElementById('v-pills-tab'), data)

            $(".nav .nav-link").on("click", function(){
                setTimeout(function() {
                        active_nav_item = $(".nav").find(".active")
                        if (active_nav_item.length > 0) {
                            selected_swarm_name = active_nav_item[0].id.split('-')[2]
                            document.getElementById('swarm-chart-column').removeAttribute('hidden');
                            document.getElementById('swarm-info-text').innerHTML = 'Simulated Devices Status ' 
                            document.getElementById('swarm-loading').style.display = '';
                            document.getElementById('graph-loading').style.display = '';

                            // var swarm_details = document.getElementById('swarmdet');
                            // while (swarm_details.firstChild) {
                            //     swarm_details.removeChild(swarm_details.firstChild);
                            // }

                            fetch('example.json')
                                .then((res)=> {
                                return res.text();
                                })
                                .then((data) => {
                                const tree = JsonView.createTree(data);
                                JsonView.render(tree, document.querySelector('.swarmdet'));
                                JsonView.collapseChildren(tree);
                                })
                                .catch((err) => {
                                console.log(err);
                            });
                        }
                    }
                )
            });
        }
    )
    
    // initialize chart
    const ctx = document.getElementById('swarm-chart').getContext('2d');
    const myChart = new Chart(ctx, {
        type: 'scatter',
        data: {
                datasets: [{
                    label: "swarm",
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1,
                    showLine: true,
                    pointRadius: 1,
                    hoverRadius: 1
                }],
            },
        options: {
            responsive: true
        }
    });


    $(function(){
        $("#swarm").load("swarm.html", null, function() {
            var table = document.getElementById("devices");
                setInterval(function() {
                    if (selected_swarm_name != null) {
                        document.getElementById('devices-table').removeAttribute('hidden');
                        document.getElementById('welcome').style.display = 'none';
                        document.getElementById('swarm-loading').style.display = 'none';
                        // update swarm info
                        getSwarmInfo(selected_swarm_name).done(function(data) {
                            appendSwarmToTable(table, data['devices']);
                        });
                    }
                    // check which swarm is currently running
                    getRunningSwarms().done(function(data) {
                        for (var i=0; i < data['runningSwarms'].length; i++) {
                            document.getElementById('spinner-'+data['runningSwarms'][i]).style.display = '';
                        }
                    });

                }, 2000);
                setInterval(function() {
                    if (selected_swarm_name != null) {
                        // update chart
                        getSwarmMetric(selected_swarm_name).done(function(data) {
                            myData = [];
                            for (var i=0; i < data['times'].length; i++) {
                                myData.push({x: data['times'][i], y: data['accs_list'][i]});
                            }
                            myChart.data.datasets[0].label = selected_swarm_name;
                            myChart.data.datasets[0].data = myData;
                            myChart.update('active');
                            document.getElementById('graph-loading').style.display = 'none';
                        });

                        
                    }
                }, 4000);
        });
    });
});

