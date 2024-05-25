import argparse
import datetime
import io
import json
import logging
import os
import threading
import requests
import signal
import sys
import zipfile
from urllib.parse import urlencode

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from ansi2html import Ansi2HTMLConverter

from flask import (
    Flask,
    session,
    url_for,
    redirect,
    render_template,
    request,
    abort,
    flash,
    send_file,
    make_response,
    jsonify,
    Response,
)
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from nebula.frontend.database import (
    list_users,
    verify,
    delete_user_from_db,
    add_user,
    update_user,
    scenario_update_record,
    scenario_set_all_status_to_finished,
    get_running_scenario,
    get_user_info,
    get_scenario_by_name,
    list_nodes_by_scenario_name,
    remove_nodes_by_scenario_name,
    get_run_hashes_scenario,
    remove_scenario_by_name,
    scenario_set_status_to_finished,
    get_all_scenarios_and_check_completed,
    check_scenario_with_role,
    update_node_record,
    save_notes,
    get_notes,
    remove_note,
)

import eventlet

eventlet.monkey_patch()
async_mode = "eventlet"

app = Flask(__name__, static_url_path='/nebula/static')
app.config["DEBUG"] = os.environ.get("NEBULA_DEBUG", False)
app.config["log_dir"] = os.environ.get("NEBULA_LOGS_DIR")
app.config["config_dir"] = os.environ.get("NEBULA_CONFIG_DIR")
app.config["cert_dir"] = os.environ.get("NEBULA_CERTS_DIR")
app.config["root_host_path"] = os.environ.get("NEBULA_ROOT_HOST")
app.config["config_frontend_dir"] = os.environ.get("FEDELLAR_CONFIG_FRONTEND_DIR", "config")
app.config["statistics_port"] = os.environ.get("NEBULA_STATISTICS_PORT", 8080)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(minutes=60)
socketio = SocketIO(
    app,
    async_mode=async_mode,
    logger=False,
    engineio_logger=False,
    cors_allowed_origins="*",
)

nodes_registration = {}


# Detect CTRL+C from parent process
def signal_handler(signal, frame):
    app.logger.info("You pressed Ctrl+C [frontend]!")
    scenario_set_all_status_to_finished()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def set_default_user():
    username = os.environ.get("NEBULA_DEFAULT_USER", "admin")
    password = os.environ.get("NEBULA_DEFAULT_PASSWORD", "admin")
    if not list_users():
        add_user(username, password, "admin")


set_default_user()


@app.errorhandler(401)
def nebula_401(error):
    return render_template("401.html"), 401


@app.errorhandler(403)
def nebula_403(error):
    return render_template("403.html"), 403


@app.errorhandler(404)
def nebula_404(error):
    return render_template("404.html"), 404


@app.errorhandler(405)
def nebula_405(error):
    return render_template("405.html"), 405


@app.errorhandler(413)
def nebula_413(error):
    return render_template("413.html"), 413


@app.template_filter("datetimeformat")
def datetimeformat(value, format="%B %d, %Y %H:%M"):
    return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S").strftime(format)


@app.route("/")
def index():
    return redirect(url_for("nebula_home"))


@app.route("/nebula")
def nebula_home():
    alerts = []
    return render_template("index.html", alerts=alerts)


@app.route("/nebula/dashboard/<scenario_name>/private")
def nebula_dashboard_private(scenario_name):
    if "user" in session.keys():
        return render_template(
            "private.html",
            scenario_name=scenario_name,
        )
    else:
        return abort(401)


@app.route("/nebula/admin")
def nebula_admin():
    if session.get("role") == "admin":
        user_list = list_users(all_info=True)
        user_table = zip(range(1, len(user_list) + 1), [user[0] for user in user_list], [user[2] for user in user_list])
        return render_template("admin.html", users=user_table)
    else:
        return abort(401)


def send_from_directory(directory, filename, **options):
    """Sends a file from a given directory with :func:`send_file`.

    :param directory: the directory to look for the file in.
    :param filename: the name of the file to send.
    :param options: the options to forward to :func:`send_file`.
    """
    return send_file(os.path.join(directory, filename), **options)


@app.route("/nebula/dashboard/<scenario_name>/save_note", methods=["POST"])
def save_note_for_scenario(scenario_name):
    if "user" in session.keys():
        data = request.get_json()
        notes = data["notes"]

        try:
            save_notes(scenario_name, notes)
            return jsonify({"status": "success"})
        except Exception as e:
            app.logger.error(e)
            return jsonify({"status": "error", "message": "Could not save the notes"}), 500
    else:
        return jsonify({"status": "error", "message": "User not logged in"}), 401


@app.route("/nebula/dashboard/<scenario_name>/notes", methods=["GET"])
def get_notes_for_scenario(scenario_name):
    notes_record = get_notes(scenario_name)
    if notes_record:
        notes_data = dict(zip(notes_record.keys(), notes_record))
        return jsonify({"status": "success", "notes": notes_data["scenario_notes"]})
    else:
        return jsonify({"status": "error", "message": "Notes not found for the specified scenario"})


@app.route("/nebula/login", methods=["POST"])
def nebula_login():
    user_submitted = request.form.get("user").upper()
    if (user_submitted in list_users()) and verify(user_submitted, request.form.get("password")):
        user_info = get_user_info(user_submitted)
        session["user"] = user_submitted
        session["role"] = user_info[2]
        return "Login successful", 200
    else:
        # flash(u'Invalid password provided', 'error')
        abort(401)


@app.route("/nebula/logout")
def nebula_logout():
    session.pop("user", None)
    return redirect(url_for("nebula_home"))


@app.route("/nebula/user/delete/<user>/", methods=["GET"])
def nebula_delete_user(user):
    if session.get("role", None) == "admin":
        if user == "ADMIN":  # ADMIN account can't be deleted.
            return abort(403)
        if user == session["user"]:  # Current user can't delete himself.
            return abort(403)

        delete_user_from_db(user)
        return redirect(url_for("nebula_admin"))
    else:
        return abort(401)


@app.route("/nebula/user/add", methods=["POST"])
def nebula_add_user():
    if session.get("role", None) == "admin":  # only Admin should be able to add user.
        # before we add the user, we need to ensure this doesn't exit in database. We also need to ensure the id is valid.
        user_list = list_users(all_info=True)
        if request.form.get("user").upper() in user_list:
            return redirect(url_for("nebula_admin"))
        elif " " in request.form.get("user") or "'" in request.form.get("user") or '"' in request.form.get("user"):
            return redirect(url_for("nebula_admin"))
        else:
            add_user(
                request.form.get("user"),
                request.form.get("password"),
                request.form.get("role"),
            )
            return redirect(url_for("nebula_admin"))
    else:
        return abort(401)


@app.route("/nebula/user/update", methods=["POST"])
def nebula_update_user():
    if session.get("role", None) == "admin":
        user = request.form.get("user")
        password = request.form.get("password")
        role = request.form.get("role")

        user_list = list_users()
        if user not in user_list:
            return redirect(url_for("nebula_admin"))
        else:
            update_user(user, password, role)
            return redirect(url_for("nebula_admin"))
    else:
        return abort(401)


@app.route("/nebula/api/dashboard", methods=["GET"])
@app.route("/nebula/dashboard", methods=["GET"])
def nebula_dashboard():
    # Get the list of scenarios
    if "user" in session.keys():
        scenarios = get_all_scenarios_and_check_completed()  # Get all scenarios after checking if they are completed
        scenario_running = get_running_scenario()
    else:
        scenarios = None
        scenario_running = None
    # Check if status of scenario_running is "completed"
    bool_completed = False
    if scenario_running:
        bool_completed = scenario_running[5] == "completed"
    if scenarios:
        if request.path == "/nebula/dashboard":
            return render_template(
                "dashboard.html",
                scenarios=scenarios,
                scenario_running=scenario_running,
                scenario_completed=bool_completed,
                user_logged_in=session.get("user"),
            )
        elif request.path == "/nebula/api/dashboard":
            scenarios_as_dict = [dict(row) for row in scenarios]
            return jsonify(scenarios_as_dict), 200
        else:
            return abort(401)

    else:
        if request.path == "/nebula/dashboard":
            return render_template("dashboard.html", user_logged_in=session.get("user"))
        elif request.path == "/nebula/api/dashboard":
            return jsonify({"scenarios_status": "not found in database"}), 200
        else:
            return abort(401)


@app.route("/nebula/api/dashboard/<scenario_name>/monitor", methods=["GET"])
@app.route("/nebula/dashboard/<scenario_name>/monitor", methods=["GET"])
def nebula_dashboard_monitor(scenario_name):
    scenario = get_scenario_by_name(scenario_name)
    if scenario:
        nodes_list = list_nodes_by_scenario_name(scenario_name)
        if nodes_list:
            # Get json data from each node configuration file
            nodes_config = []
            # Generate an array with True for each node that is running
            nodes_status = []
            for i, node in enumerate(nodes_list):
                nodes_config.append((node[2], node[3], node[4]))  # IP, Port, Role
                if datetime.datetime.now() - datetime.datetime.strptime(node[8], "%Y-%m-%d %H:%M:%S.%f") > datetime.timedelta(seconds=25):
                    nodes_status.append(False)
                else:
                    nodes_status.append(True)
            nodes_table = zip(
                [x[0] for x in nodes_list],  # UID
                [x[1] for x in nodes_list],  # IDX
                [x[2] for x in nodes_list],  # IP
                [x[3] for x in nodes_list],  # Port
                [x[4] for x in nodes_list],  # Role
                [x[5] for x in nodes_list],  # Neighbors
                [x[6] for x in nodes_list],  # Latitude
                [x[7] for x in nodes_list],  # Longitude
                [x[8] for x in nodes_list],  # Timestamp
                [x[9] for x in nodes_list],  # Federation
                [x[10] for x in nodes_list],  # Round
                [x[11] for x in nodes_list],  # Scenario name
                [x[12] for x in nodes_list],  # Run hash
                nodes_status,  # Status
            )

            if os.path.exists(os.path.join(app.config["config_dir"], scenario_name, "topology.png")):
                if os.path.getmtime(os.path.join(app.config["config_dir"], scenario_name, "topology.png")) < max(
                    [
                        os.path.getmtime(
                            os.path.join(
                                app.config["config_dir"],
                                scenario_name,
                                f"participant_{node[1]}.json",
                            )
                        )
                        for node in nodes_list
                    ]
                ):
                    # Update the 3D topology and image
                    update_topology(scenario[0], nodes_list, nodes_config)
            else:
                update_topology(scenario[0], nodes_list, nodes_config)

            if request.path == "/nebula/dashboard/" + scenario_name + "/nebula/monitor":
                return render_template("monitor.html", scenario_name=scenario_name, scenario=scenario, nodes=nodes_table, user_logged_in=session.get("user"))
            elif request.path == "/nebula/api/dashboard/" + scenario_name + "/nebula/monitor":
                return (
                    jsonify(
                        {
                            "scenario_status": scenario[5],
                            "nodes_table": list(nodes_table),
                            "scenario_name": scenario[0],
                            "scenario_title": scenario[3],
                            "scenario_description": scenario[4],
                        }
                    ),
                    200,
                )
            else:
                return abort(401)
        else:
            # There is a scenario but no nodes
            if request.path == "/nebula/dashboard/" + scenario_name + "/nebula/monitor":
                return render_template("monitor.html", scenario_name=scenario_name, scenario=scenario, nodes=[], user_logged_in=session.get("user"))
            elif request.path == "/nebula/api/dashboard/" + scenario_name + "/nebula/monitor":
                return (
                    jsonify(
                        {
                            "scenario_status": scenario[5],
                            "nodes_table": [],
                            "scenario_name": scenario[0],
                            "scenario_title": scenario[3],
                            "scenario_description": scenario[4],
                        }
                    ),
                    200,
                )
            else:
                return abort(401)
    else:
        # There is no scenario
        if request.path == "/nebula/dashboard/" + scenario_name + "/nebula/monitor":
            return render_template("monitor.html", scenario_name=scenario_name, scenario=None, nodes=[], user_logged_in=session.get("user"))
        elif request.path == "/nebula/api/dashboard/" + scenario_name + "/nebula/monitor":
            return jsonify({"scenario_status": "not exists"}), 200
        else:
            return abort(401)


def update_topology(scenario_name, nodes_list, nodes_config):
    import numpy as np

    nodes = []
    for node in nodes_list:
        nodes.append(node[2] + ":" + str(node[3]))
    matrix = np.zeros((len(nodes), len(nodes)))
    for node in nodes_list:
        for neighbour in node[5].split(" "):
            if neighbour != "":
                if neighbour in nodes:
                    matrix[
                        nodes.index(node[2] + ":" + str(node[3])),
                        nodes.index(neighbour),
                    ] = 1
    from nebula.addons.topologymanager import TopologyManager

    tm = TopologyManager(n_nodes=len(nodes_list), topology=matrix, scenario_name=scenario_name)
    tm.update_nodes(nodes_config)
    tm.draw_graph(path=os.path.join(app.config["config_dir"], scenario_name, f"topology.png"))


@app.route("/nebula/dashboard/<scenario_name>/node/update", methods=["POST"])
def nebula_update_node(scenario_name):
    if request.method == "POST":
        # Check if the post request is a json, if not, return 400
        if request.is_json:
            config = request.get_json()
            timestamp = datetime.datetime.now()
            # Update the node in database
            update_node_record(
                str(config["device_args"]["uid"]),
                str(config["device_args"]["idx"]),
                str(config["network_args"]["ip"]),
                str(config["network_args"]["port"]),
                str(config["device_args"]["role"]),
                str(config["network_args"]["neighbors"]),
                str(config["mobility_args"]["latitude"]),
                str(config["mobility_args"]["longitude"]),
                str(timestamp),
                str(config["scenario_args"]["federation"]),
                str(config["federation_args"]["round"]),
                str(config["scenario_args"]["name"]),
                str(config["tracking_args"]["run_hash"]),
            )

            neighbors_distance = config["mobility_args"]["neighbors_distance"]

            node_update = {
                "scenario_name": scenario_name,
                "uid": config["device_args"]["uid"],
                "idx": config["device_args"]["idx"],
                "ip": config["network_args"]["ip"],
                "port": str(config["network_args"]["port"]),
                "role": config["device_args"]["role"],
                "neighbors": config["network_args"]["neighbors"],
                "latitude": config["mobility_args"]["latitude"],
                "longitude": config["mobility_args"]["longitude"],
                "timestamp": str(timestamp),
                "federation": config["scenario_args"]["federation"],
                "round": config["federation_args"]["round"],
                "name": config["scenario_args"]["name"],
                "status": True,
                "neighbors_distance": neighbors_distance,
            }

            # Send notification to each connected users
            try:
                socketio.emit("node_update", node_update)
            except Exception as e:
                app.logger.error(f"Error sending node_update to socketio: {e}")
                pass

            # Return only the code 200
            return jsonify({"message": "Node updated", "status": "success"}), 200

        else:
            return abort(400)


@app.route("/nebula/dashboard/<scenario_name>/node/register", methods=["POST"])
def nebula_register_node(scenario_name):
    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
            node = data["node"]
            with nodes_registration[scenario_name]["condition"]:
                nodes_registration[scenario_name]["nodes"].add(node)
                app.logger.info(f"Node {node} registered")
                if len(nodes_registration[scenario_name]["nodes"]) == nodes_registration[scenario_name]["n_nodes"]:
                    nodes_registration[scenario_name]["condition"].notify_all()
                    app.logger.info("All nodes registered")

            # Return only the code 200
            return jsonify({"message": "Node registered", "status": "success"}), 200

        else:
            return abort(400)


@app.route("/nebula/dashboard/<scenario_name>/node/wait", methods=["GET"])
def nebula_wait_nodes(scenario_name):
    if scenario_name not in nodes_registration:
        return jsonify({"message": "Scenario not found", "status": "error"}), 404

    with nodes_registration[scenario_name]["condition"]:
        while len(nodes_registration[scenario_name]["nodes"]) < nodes_registration[scenario_name]["n_nodes"]:
            nodes_registration[scenario_name]["condition"].wait()
        return jsonify({"message": "All nodes registered", "status": "success"}), 200


@app.route("/nebula/dashboard/<scenario_name>/node/<id>/infolog", methods=["GET"])
def nebula_monitor_log(scenario_name, id):
    logs = os.path.join(app.config["log_dir"], scenario_name, f"participant_{id}.log")
    if os.path.exists(logs):
        return send_file(logs, mimetype="text/plain", as_attachment=True)
    else:
        abort(404)


@app.route("/nebula/dashboard/<scenario_name>/node/<id>/infolog/<number>", methods=["GET"])
def nebula_monitor_log_x(scenario_name, id, number):
    # Send file (is not a json file) with the log
    logs = os.path.join(app.config["log_dir"], scenario_name, f"participant_{id}.log")
    if os.path.exists(logs):
        # Open file maintaining the file format
        with open(logs, "r") as f:
            # Read the last n lines of the file
            lines = f.readlines()[-int(number) :]
            # Join the lines in a single string
            lines = "".join(lines)
            # Convert the ANSI escape codes to HTML
            converter = Ansi2HTMLConverter()
            html_text = converter.convert(lines, full=False)
            # Return the string
            return Response(html_text, mimetype="text/plain")
    else:
        return Response("No logs available", mimetype="text/plain")


@app.route("/nebula/dashboard/<scenario_name>/node/<id>/debuglog", methods=["GET"])
def nebula_monitor_log_debug(scenario_name, id):
    logs = os.path.join(app.config["log_dir"], scenario_name, f"participant_{id}_debug.log")
    if os.path.exists(logs):
        return send_file(logs, mimetype="text/plain", as_attachment=True)
    else:
        abort(404)


@app.route("/nebula/dashboard/<scenario_name>/node/<id>/errorlog", methods=["GET"])
def nebula_monitor_log_error(scenario_name, id):
    logs = os.path.join(app.config["log_dir"], scenario_name, f"participant_{id}_error.log")
    if os.path.exists(logs):
        return send_file(logs, mimetype="text/plain", as_attachment=True)
    else:
        abort(404)


@app.route("/nebula/dashboard/<scenario_name>/topology/image/", methods=["GET"])
def nebula_monitor_image(scenario_name):
    topology_image = os.path.join(app.config["config_dir"], scenario_name, f"topology.png")
    if os.path.exists(topology_image):
        return send_file(topology_image, mimetype="image/png")
    else:
        abort(404)


def stop_scenario(scenario_name):
    from nebula.controller import Controller

    Controller.stop_participants()
    Controller.stop_blockchain()
    scenario_set_status_to_finished(scenario_name)


def stop_all_scenarios():
    from nebula.controller import Controller

    Controller.stop_participants()
    Controller.stop_blockchain()
    scenario_set_all_status_to_finished()


@app.route("/nebula/dashboard/<scenario_name>/stop", methods=["GET"])
def nebula_stop_scenario(scenario_name):
    # Stop the scenario
    if "user" in session.keys():
        if session["role"] == "demo":
            return abort(401)
        elif session["role"] == "user":
            if not check_scenario_with_role(session["role"], scenario_name):
                return abort(401)
        stop_scenario(scenario_name)
        return redirect(url_for("nebula_dashboard"))
    else:
        return abort(401)


def remove_scenario(scenario_name=None):
    from nebula.controller import Controller

    # TODO: AIM START
    # try:
    #     repo = Repo.from_path(f"{app.config['log_dir']}")
    #     list_tuples_participant_hash = get_run_hashes_scenario(scenario_name)
    #     hashes = [tuple[1] for tuple in list_tuples_participant_hash]
    #     app.logger.info(f"Removing statistics from {scenario_name}: {hashes}")
    #     for hash in hashes:
    #         run = repo.get_run(hash)
    #         run.close()
    #     success, remaining_runs = repo.delete_runs(hashes)
    #     if success:
    #         app.logger.info(f'Successfully deleted {len(hashes)} runs.')
    #     else:
    #         app.logger.info('Something went wrong while deleting runs.')
    #         app.logger.info(f'Remaining runs: {remaining_runs}')
    # except Exception as e:
    #     app.logger.error(f"Error removing statistics from {scenario_name}: {e}")
    #     pass

    # Remove all files and folders (recursively) which contain the previous hashes in the name
    # for hash in hashes:
    #     Controller.remove_files_by_run_hash(hash)
    # TODO: AIM END
    # Remove registered nodes and conditions
    nodes_registration.pop(scenario_name, None)
    remove_nodes_by_scenario_name(scenario_name)
    remove_scenario_by_name(scenario_name)
    remove_note(scenario_name)
    Controller.remove_files_by_scenario(scenario_name)


@app.route("/nebula/dashboard/<scenario_name>/remove", methods=["GET"])
def nebula_remove_scenario(scenario_name):
    # Remove the scenario
    if "user" in session.keys():
        if session["role"] == "demo":
            return abort(401)
        elif session["role"] == "user":
            if not check_scenario_with_role(session["role"], scenario_name):
                return abort(401)
        remove_scenario(scenario_name)
        return redirect(url_for("nebula_dashboard"))
    else:
        return abort(401)


# TODO: AIM START

# def get_tracking_hash_scenario(scenario_name):
#     import requests
#     url = f"http://127.0.0.1:{app.config['statistics_port']}/statistics/api/experiments"
#     # Get JSON data from the URL
#     response = requests.get(url)
#     if response.status_code == 200:
#         experiments = response.json()
#         for experiment in experiments:
#             if experiment["name"] == scenario_name:
#                 return experiment["id"]

#     return None

# @app.route("/nebula/dashboard/statistics/", methods=["GET"])
# @app.route("/nebula/dashboard/<scenario_name>/statistics/", methods=["GET"])
# def nebula_dashboard_statistics(scenario_name=None):
#     # Adjust the filter to the scenario name
#     statistics_url = "/nebula/statistics/"
#     if scenario_name is not None:
#         experiment_hash = get_tracking_hash_scenario(scenario_name=scenario_name)
#         statistics_url += f"experiments/{experiment_hash}/runs"

#     return render_template("statistics.html", statistics_url=statistics_url)

# @app.route("/nebula/dashboard/<scenario_name>/node/<hash>/metrics", methods=["GET"])
# def nebula_dashboard_node_metrics(scenario_name, hash):
#     # Get the metrics of a node
#     statistics_url = f"/nebula/statistics/runs/{hash}/metrics"
#     return render_template("statistics.html", statistics_url=statistics_url)

# @app.route("/nebula/statistics/", methods=["GET", "POST"])
# @app.route("/nebula/statistics/<path:path>", methods=["GET", "POST"])
# def statistics_proxy(path=None):
#     if "user" in session.keys():
#         query_string = urlencode(request.args)

#         url = f"http://127.0.0.1:{app.config['statistics_port']}/statistics"

#         url = f"{url}{('/' + path) if path else '/'}" + (
#             "?" + query_string if query_string else ""
#         )

#         response = requests.request(
#             method=request.method,
#             url=url,
#             headers={key: value for (key, value) in request.headers if key != "Host"},
#             data=request.get_data(),
#             cookies=request.cookies,
#             allow_redirects=False,
#         )

#         excluded_headers = [
#             "content-encoding",
#             "content-length",
#             "transfer-encoding",
#             "connection",
#         ]
#         headers = [
#             (name, value)
#             for (name, value) in response.raw.headers.items()
#             if name.lower() not in excluded_headers
#         ]

#         response = Response(response.content, response.status_code, headers)
#         return response
#     else:
#         return abort(401)

# @app.route("/nebula/dashboard/<scenario_name>/download/metrics", methods=["GET"])
# def nebula_dashboard_download_metrics(scenario_name):
#     if "user" in session.keys():
#         # Get the metrics of the scenario
#         os.makedirs(os.path.join(app.config["log_dir"], scenario_name, "metrics"), exist_ok=True)

#         aim_repo = Repo.from_path("/nebula/nebula/app/logs")
#         query = "run.experiment == '{}'".format(scenario_name)
#         df = aim_repo.query_metrics(query).dataframe()

#         hash_to_participant = {hash: participant for participant, hash in get_run_hashes_scenario(scenario_name)}
#         df["participant"] = df["run.hash"].map(hash_to_participant)
#         df.drop(columns=["run", "run.hash", "metric.context", "epoch"], axis=1, inplace=True)
#         cols = df.columns.tolist()
#         cols.remove("participant")
#         cols.remove("metric.name")
#         df = df.reindex(columns=["participant", "metric.name"] + cols)

#         for name, group in df.groupby("participant"):
#             group.to_csv(
#                 os.path.join(app.config["log_dir"], scenario_name, "metrics", f"{name}.csv"),
#                 index=True,
#             )

#         # Create a zip file with the metrics, send it to the user and delete it
#         memory_file = io.BytesIO()
#         with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zipf:
#             zipdir(os.path.join(app.config["log_dir"], scenario_name, "metrics"), zipf)

#         memory_file.seek(0)

#         return send_file(
#             memory_file,
#             mimetype="application/zip",
#             as_attachment=True,
#             download_name=f"{scenario_name}_metrics.zip",
#         )
#     else:
#         return abort(401)

# TODO: AIM END


# TODO: TENSORBOARD START
@app.route("/nebula/dashboard/statistics/", methods=["GET"])
@app.route("/nebula/dashboard/<scenario_name>/statistics/", methods=["GET"])
def nebula_dashboard_statistics(scenario_name=None):
    # Adjust the filter to the scenario name
    tensorboard_url = "/nebula/statistics/"
    if scenario_name is not None:
        tensorboard_url += f"?smoothing=0&runFilter={scenario_name}"

    return render_template("statistics.html", statistics_url=tensorboard_url)


@app.route("/nebula/statistics/", methods=["GET", "POST"])
@app.route("/nebula/statistics/<path:path>", methods=["GET", "POST"])
def statistics_proxy(path=None):
    import requests

    query_string = urlencode(request.args)
    # Internal port of statistics service
    url = f"http://localhost:8080"

    tensorboard_url = f"{url}{('/' + path) if path else ''}" + ("?" + query_string if query_string else "")

    response = requests.request(
        method=request.method,
        url=tensorboard_url,
        headers={key: value for (key, value) in request.headers if key != "Host"},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
    )

    excluded_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]
    headers = [(name, value) for (name, value) in response.raw.headers.items() if name.lower() not in excluded_headers]

    if "text/html" in response.headers["Content-Type"]:
        # Replace the resources URLs to point to the proxy
        content = response.text
        content = content.replace("url(/", f"url(/statistics/")
        response = Response(content, response.status_code, headers)
        return response

    # Construye y envía la respuesta
    response = Response(response.content, response.status_code, headers)
    return response


@app.route("/nebula/experiment/<path:path>", methods=["GET", "POST"])
def experiment_proxy(path=None):
    query_string = request.query_string.decode("utf-8")
    new_url = url_for("statistics_proxy", path="experiment/" + path)
    if query_string:
        new_url += "?" + query_string

    return redirect(new_url)


# TODO: TENSORBOARD END


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )


@app.route("/nebula/dashboard/<scenario_name>/download/logs", methods=["GET"])
def nebula_dashboard_download_logs_metrics(scenario_name):
    if "user" in session.keys():
        log_folder = os.path.join(app.config["log_dir"], scenario_name)
        config_folder = os.path.join(app.config["config_dir"], scenario_name)
        if os.path.exists(log_folder) and os.path.exists(config_folder):
            # Create a zip file with the logs and the config files, send it to the user and delete it
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipdir(log_folder, zipf)
                zipdir(config_folder, zipf)

            memory_file.seek(0)

            return send_file(
                memory_file,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"{scenario_name}.zip",
            )
    else:
        return abort(401)


@app.route("/nebula/dashboard/deployment/", methods=["GET"])
def nebula_dashboard_deployment():
    scenario_running = get_running_scenario()
    return render_template("deployment.html", scenario_running=scenario_running, user_logged_in=session.get("user"))


def attack_node_assign(
    nodes,
    federation,
    attack,
    poisoned_node_percent,
    poisoned_sample_percent,
    poisoned_noise_percent,
):
    """Identify which nodes will be attacked"""
    import random
    import math

    attack_matrix = []
    n_nodes = len(nodes)
    if n_nodes == 0:
        return attack_matrix

    nodes_index = []
    # Get the nodes index
    if federation == "DFL":
        nodes_index = list(nodes.keys())
    else:
        for node in nodes:
            if nodes[node]["role"] != "server":
                nodes_index.append(node)

    n_nodes = len(nodes_index)
    # Number of attacked nodes, round up
    num_attacked = int(math.ceil(poisoned_node_percent / 100 * n_nodes))
    if num_attacked > n_nodes:
        num_attacked = n_nodes

    # Get the index of attacked nodes
    attacked_nodes = random.sample(nodes_index, num_attacked)

    # Assign the role of each node
    for node in nodes:
        node_att = "No Attack"
        attack_sample_persent = 0
        poisoned_ratio = 0
        if (node in attacked_nodes) or (nodes[node]["malicious"]):
            node_att = attack
            attack_sample_persent = poisoned_sample_percent / 100
            poisoned_ratio = poisoned_noise_percent / 100
        nodes[node]["attacks"] = node_att
        nodes[node]["poisoned_sample_percent"] = attack_sample_persent
        nodes[node]["poisoned_ratio"] = poisoned_ratio
        attack_matrix.append([node, node_att, attack_sample_persent, poisoned_ratio])
    return nodes, attack_matrix


import math


def mobility_assign(nodes, mobile_participants_percent):
    """Assign mobility to nodes"""
    import random

    # Number of mobile nodes, round down
    num_mobile = math.floor(mobile_participants_percent / 100 * len(nodes))
    if num_mobile > len(nodes):
        num_mobile = len(nodes)

    # Get the index of mobile nodes
    mobile_nodes = random.sample(list(nodes.keys()), num_mobile)

    # Assign the role of each node
    for node in nodes:
        node_mob = False
        if node in mobile_nodes:
            node_mob = True
        nodes[node]["mobility"] = node_mob
    return nodes


@app.route("/nebula/dashboard/deployment/run", methods=["POST"])
def nebula_dashboard_deployment_run():
    from nebula.controller import Controller

    if "user" in session.keys():
        if session["role"] == "demo":
            return abort(401)
        elif session["role"] == "user":
            # If there is a scenario running, abort
            if get_running_scenario():
                return abort(401)
        # Receive a JSON data with the scenario configuration
        if request.is_json:
            # Stop the running scenario
            stop_all_scenarios()
            data = request.get_json()
            nodes = data["nodes"]
            scenario_name = f'nebula_{data["federation"]}_{datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'

            scenario_path = os.path.join(app.config["config_dir"], scenario_name)
            os.makedirs(scenario_path, exist_ok=True)

            scenario_file = os.path.join(scenario_path, "scenario.json")
            with open(scenario_file, "w") as f:
                json.dump(data, f, sort_keys=False, indent=2)

            args_controller = {
                "scenario_name": scenario_name,
                "config": app.config["config_dir"],
                "logs": app.config["log_dir"],
                "certs": app.config["cert_dir"],
                "n_nodes": data["n_nodes"],
                "matrix": data["matrix"],
                "federation": data["federation"],
                "topology": data["topology"],
                "simulation": data["simulation"],
                "env": None,
                "root_path": app.config["root_host_path"],
                "webport": request.host.split(":")[1] if ":" in request.host else 80,  # Get the port of the frontend, if not specified, use 80
                "network_subnet": data["network_subnet"],
                "use_blockchain": data["agg_algorithm"] == "BlockchainReputation",
                "network_gateway": data["network_gateway"],
            }
            # Save args in a file
            controller_file = os.path.join(app.config["config_dir"], scenario_name, "controller.json")
            with open(controller_file, "w") as f:
                json.dump(args_controller, f, sort_keys=False, indent=2)

            # Get attack info
            attack = data["attacks"]
            poisoned_node_percent = int(data["poisoned_node_percent"])
            poisoned_sample_percent = int(data["poisoned_sample_percent"])
            poisoned_noise_percent = int(data["poisoned_noise_percent"])
            federation = data["federation"]
            # Get attack matrix
            nodes, attack_matrix = attack_node_assign(
                nodes,
                federation,
                attack,
                poisoned_node_percent,
                poisoned_sample_percent,
                poisoned_noise_percent,
            )

            mobility_status = data["mobility"]
            if mobility_status:
                # Mobility parameters (selecting mobile nodes)
                mobile_participants_percent = int(data["mobile_participants_percent"])
                # Assign mobility to nodes depending on the percentage
                nodes = mobility_assign(nodes, mobile_participants_percent)
            else:
                # Assign mobility to nodes depending on the percentage
                nodes = mobility_assign(nodes, 0)

            # For each node, create a new file in config directory
            import shutil

            # Loop dictionary of nodes
            for node in nodes:
                node_config = nodes[node]
                # Create a copy of participant.json.example and update the file with the update values
                participant_file = os.path.join(
                    app.config["config_dir"],
                    scenario_name,
                    f'participant_{node_config["id"]}.json',
                )
                os.makedirs(os.path.dirname(participant_file), exist_ok=True)
                # Create a copy of participant.json.example
                shutil.copy(
                    os.path.join(
                        app.config["config_frontend_dir"],
                        f"participant.json.example",
                    ),
                    participant_file,
                )
                # Update IP, port, and role
                with open(participant_file) as f:
                    participant_config = json.load(f)
                participant_config["network_args"]["ip"] = node_config["ip"]
                participant_config["network_args"]["port"] = int(node_config["port"])
                participant_config["device_args"]["idx"] = node_config["id"]
                participant_config["device_args"]["start"] = node_config["start"]
                participant_config["device_args"]["role"] = node_config["role"]
                participant_config["device_args"]["proxy"] = node_config["proxy"]
                participant_config["device_args"]["malicious"] = node_config["malicious"]
                participant_config["scenario_args"]["rounds"] = int(data["rounds"])
                participant_config["data_args"]["dataset"] = data["dataset"]
                participant_config["data_args"]["iid"] = data["iid"]
                participant_config["data_args"]["partition_selection"] = data["partition_selection"]
                participant_config["data_args"]["partition_parameter"] = data["partition_parameter"]
                participant_config["model_args"]["model"] = data["model"]
                participant_config["training_args"]["epochs"] = int(data["epochs"])
                participant_config["device_args"]["accelerator"] = data["accelerator"]
                participant_config["device_args"]["logging"] = data["logginglevel"]
                participant_config["aggregator_args"]["algorithm"] = data["agg_algorithm"]

                participant_config["adversarial_args"]["attacks"] = node_config["attacks"]
                participant_config["adversarial_args"]["poisoned_sample_percent"] = node_config["poisoned_sample_percent"]
                participant_config["adversarial_args"]["poisoned_ratio"] = node_config["poisoned_ratio"]
                participant_config["defense_args"]["with_reputation"] = data["with_reputation"]
                participant_config["defense_args"]["is_dynamic_topology"] = data["is_dynamic_topology"]
                participant_config["defense_args"]["is_dynamic_aggregation"] = data["is_dynamic_aggregation"]
                participant_config["defense_args"]["target_aggregation"] = data["target_aggregation"]

                participant_config["mobility_args"]["random_geo"] = data["random_geo"]
                participant_config["mobility_args"]["latitude"] = data["latitude"]
                participant_config["mobility_args"]["longitude"] = data["longitude"]
                # Get mobility config for each node (after applying the percentage from the frontend)
                participant_config["mobility_args"]["mobility"] = node_config["mobility"]
                participant_config["mobility_args"]["mobility_type"] = data["mobility_type"]
                participant_config["mobility_args"]["radius_federation"] = data["radius_federation"]
                participant_config["mobility_args"]["scheme_mobility"] = data["scheme_mobility"]
                participant_config["mobility_args"]["round_frequency"] = data["round_frequency"]

                with open(participant_file, "w") as f:
                    json.dump(participant_config, f, sort_keys=False, indent=2)

            # Create a argparse object
            import argparse
            import subprocess

            args_controller = argparse.Namespace(**args_controller)
            controller = Controller(args_controller)  # Generate an instance of controller in this new process
            try:
                if mobility_status:
                    additional_participants = data["additional_participants"]  # List of additional participants with dict("round": int). Example: [{"round": 1}, {"round": 2}]
                    schema_additional_participants = data["schema_additional_participants"]
                    controller.load_configurations_and_start_nodes(additional_participants, schema_additional_participants)
                else:
                    controller.load_configurations_and_start_nodes()
            except subprocess.CalledProcessError as e:
                app.logger.error(f"Error docker-compose up: {e}")
                return redirect(url_for("nebula_dashboard_deployment"))
            # Generate/Update the scenario in the database
            scenario_update_record(
                scenario_name=controller.scenario_name,
                start_time=controller.start_date_scenario,
                end_time="",
                status="running",
                title=data["scenario_title"],
                description=data["scenario_description"],
                network_subnet=data["network_subnet"],
                model=data["model"],
                dataset=data["dataset"],
                rounds=data["rounds"],
                role=session["role"],
            )
            # Update nodes_ready variable
            nodes_registration[scenario_name] = {
                "n_nodes": data["n_nodes"],
                "nodes": set(),
            }
            nodes_registration[scenario_name]["condition"] = threading.Condition()
            return redirect(url_for("nebula_dashboard"))
        else:
            return abort(401)
    else:
        return abort(401)


if __name__ == "__main__":
    # Parse args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port to run the frontend on.")
    args = parser.parse_args()
    app.logger.info(f"Starting frontend on port {args.port}")
    # app.run(debug=True, host="0.0.0.0", port=int(args.port))
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    # Get env variables
    socketio.run(app, debug=True, host="0.0.0.0", port=int(args.port))
