### How to run?
* Create a folder `./data`.
* Create folder `./data/{network_name}`.
* Add `{network_name}_net.tntp` and `{network_name}_trips.tntp` files to
    `./data/{network_name}/`
* Create folder `./results`. 
* Create subfolders `./results/{network_name}` for each test network.
* Run `python3 main.py` to generate data for
    the algorithm in `./results/network_name/`
* Generate plot by running `python3 plotter.py {network_name}`
    (this creates the plot in `./results/{network_name}.png`)
