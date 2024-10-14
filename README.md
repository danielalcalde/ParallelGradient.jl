Depending on your workload you can use the standart pmap function as a backend or the pmap_chuncked function. The pmap_chuncked function is a wrapper around the pmap function that splits the input data into chunks first and then starts one job per worker.

TODO: implement tmap_chunck