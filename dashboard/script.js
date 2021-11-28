
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
//['DeviceProgress', 'N'],
    var key_type = [['DeviceStatus', 'S'], ['EncIdx', 'N'], ['Hostname', 'S']];
    for (var i = 0; i < key_type.length; i++) {
        row_html += '<td>';
        if (key_type[i][0] == 'Hostname') {
            row_html += data[key_type[i][0]][key_type[i][1]].split('-')[3];
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
    table.appendChild(row)
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
                            console.log(selected_swarm_name) //@TODO call when rendering swarm page
                            document.getElementById('swarm-info-text').innerHTML = 'Simulated Devices Status';
                        }
                    }
                )
            });
        }
    )

    $(function(){
        $("#swarm").load("swarm.html", null, function() {
            var table = document.getElementById("devices");
                setInterval(function() {
                    if (selected_swarm_name != null) {
                        getSwarmInfo(selected_swarm_name).done(function(data) {
                            appendSwarmToTable(table, data['devices']);
                        });
                    }

                }, 2000);
        });
    });

    var data = [{
        id: 1,
        status: 'Item 1',
        progress: 'good'
    }, {
        id: 2,
        status: 'Item 2',
        progress: 'bad'
    }]; 

    
});