/**
 * Implements web based all - relations service.
 */

tf.setBackend('cpu')

// this holds the state of the application
var current_results = null

/**
 * Resets the visualization of results with
 * most recent results, stored in current_results
 * variable in this script.
 */
function reset_csv_table(){
    // produce sorted concepts
    var concepts = []

    // get column headers
    for(var c in current_results){
        concepts.push(c)
    }
    concepts.sort()

    var data = [[" "]]

    // headers are added here
    for(var c of concepts){
        data[0].push(c)
    }

    // generate all pairs of concepts
    for(var A of concepts){
        var row = ["from " + A]
        for(var B of concepts){
            var weight = current_results[A][B]
            if(A !== B){
                weight = weight.toFixed(2)
            }
            row.push(weight)
        }
        data.push(row)
    }
    

    // calculate the sizes of columns
    var autoWidths = [0] 
    var mxWd = 0
    var char_size = 9
    for(var c of concepts){
        var column_size = (3+c.length)*char_size
        autoWidths.push(column_size)
        if(mxWd < column_size){
            mxWd = column_size
        }
    }
    autoWidths[0] = mxWd + 4*char_size // account for "from"

    // render the CSV
    $('#relation_table').jexcel({ data:data, colWidths: autoWidths});
}

/**
 * Updates the graph of the results with specified
 * threshold by the user interface.
 */
function reset_graph(){
    if(current_results === null){
        return
    }

    // get current threshold
    var threshold = document.getElementById('relation_threshold').value / 100.0
    document.getElementById('threshold_value').innerHTML = threshold.toFixed(2)
    var nodes_edges = []

    // add all the edges
    for(var A in current_results){
        nodes_edges.push({
            data: {id: A, label: A}, classes: 'top-left'
        })
    }

    // necessary for the layout of the graph
    var n_rows = Math.ceil(Math.sqrt(nodes_edges.length))

    // select all the relations that are above the threshold
    for(var A in current_results){
        for(var B in current_results){
            if(A === B){
                continue
            }

            var weight = current_results[A][B]

            // skip weights with too small value
            if(weight < threshold){
                continue
            }

            nodes_edges.push({
                data: {label: weight.toFixed(2), source: A, target: B}
            })
        }
    }

    // run the visualization library
    var cy = cytoscape({
        container: document.getElementById('cy'),
        elements: nodes_edges,
        style: [ // the stylesheet for the graph
            {
                selector: 'node',
                style: {
                    'background-color': '#666',
                    'label': 'data(label)'
                }
            },
            {
                selector: 'edge',
                style: {
                    'label': 'data(label)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'width': 3,
                    'line-color': '#bbb',
                    'target-arrow-color': '#bbb'
                }
            }
        ],

        layout: {
            name: 'grid',
            rows: n_rows
        }
    });     
}

 /**
  * Shows the message to the user.
  * @param {string} message message to be recorded.
  */
function log_progress(message, type){

    if(type === "relations"){
        var relation_log = document.getElementById('relations_log')
        relation_log.value += message + "\n"
        relation_log.scrollTop = relation_log.scrollHeight;
    }else if(type === "models"){
        var model_log = document.getElementById('models_log')
        model_log.value += message + "\n"
        model_log.scrollTop = model_log.scrollHeight;
    }else if(type === "final"){
        current_results = message
        reset_csv_table()
        reset_graph()
    }else{
        console.log(message)
    }
}

/**
 * Parse the CSV file into set of concepts.
 * @param {object} results Representation of the CSV file
 */
function extract_concepts(results){
    var fields = results['meta']['fields']

    // extract the concepts, initialize with empty array
    // which will be filled with feature vectors.
    var concepts = {}
    var concept_map = {}
    for(var f of fields){
        var concept = f.split('_')[0]
        concepts[concept] = []
        concept_map[f] = concept
    }
    var new_feats = {}
    // sort row values into different concepts
    for(var row of results['data']){
        // add a new feature vector to every concept
        
        for(var c in concepts){
            new_feats[c] = []
        }

        // sort the fields into concepts
        for(var f of fields){
            var value = Number.parseFloat(row[f])
            new_feats[concept_map[f]].push(value)
        }

        // append the newly generated vectors
        for(var c in new_feats){
            concepts[c].push(new_feats[c])
        }
    }

    return concepts
}

/**
 * Split data into training and testing.
 * @param {Array} X Input samples
 * @param {Array} y Output samples
 * @param {Number} test_split Fraction of data to be used for testing
 */
function train_test_split(X, y, test_split=0.3){
    var I = X.map((v, i)=>Math.random()>test_split)

    var result = []

    for(var a of [X, y]){
        result.push(a.filter((v,i)=>I[i])) // train partition
        result.push(a.filter((v,i)=>!I[i])) // test partition
    }
    
    return result
}

/**
 * Returns the coefficient of determination for the 
 * two arrays.
 * @param {Array} y_true True values
 * @param {Array} y_pred Estimated values
 */
function r2_score(y_true, y_pred){    
    var mean = tf.mean(y_true) // can be seen as predictions of a constant model
    
    // calculate errors of trivial model
    var base_errors = tf.sub(y_true, mean)
    base_errors = tf.sum(tf.pow(base_errors, 2))
    base_errors = base_errors.get()

    var model_errors = tf.sub(y_true, y_pred)
    model_errors = tf.sum(tf.pow(model_errors, 2))
    model_errors = model_errors.get()

    if(base_errors === 0.0){
        return 0.0
    }

    return 1.0 - model_errors / base_errors
}

class SGDRegressor{
    constructor(params){
        this.params = params
        this.model = null

        this.scale = null
        this.scale_y = null
    }

    async transform_X(X){
        if(this.scale === null){
            // calculate initial statistics
            this.mean = tf.mean(X, 0)
            var variance = tf.mean(tf.pow(tf.abs(tf.sub(X, this.mean)),2), 0)
            this.scale = tf.sqrt(variance)
        }

        return X.sub(this.mean).div(this.scale)
    }

    async transform_y(y){
        if(this.scale_y === null){
            this.mean_y = tf.mean(y)
            this.scale_y = tf.max(tf.abs(y.sub(this.mean_y)))
        }

        return y.sub(this.mean_y).div(this.scale_y)
    }

    async inverse_t_y(y){
        return y.mul(this.scale_y).add(this.mean_y)
    }

    async fit(X, y){
        var X = tf.tensor2d(X)
        var y = tf.tensor1d(y)

        // preprocess data for numeric stability
        X = await this.transform_X(X)
        y = await this.transform_y(y)

        var l1_ratio = this.params['l1_ratio']
        var alpha = this.params['alpha']
        var learning_rate = this.params['eta0']
        var epochs = this.params['max_iter']
        var batch_size = this.params['batch_size']

        var sample_shape = X.shape.slice(1)
        
        var l1_alpha = l1_ratio*alpha
        var l2_alpha = (1.0 - l1_ratio)*alpha
        var regularization = tf.regularizers.l1l2({
            'l1': l1_alpha,
            'l2': l2_alpha
        })

        var model = tf.sequential()
        model.add(tf.layers.dense({
            units: 1, 
            inputShape: sample_shape, 
            biasRegularizer: regularization, 
            kernelRegularizer: regularization
        }))
        
        await model.compile({
            loss: "meanSquaredError", 
            optimizer: tf.train.adam(learning_rate)
        });

        await model.fit(X, y, {'epochs': epochs, 'batchSize': batch_size})

        this.model = model
    }

    async predict(X){
        var model = this.model
        X = await this.transform_X(X)
        var y_pred = await model.predict(X)

        // drop an extra dimension
        y_pred = tf.reshape(y_pred, [-1])

        // bring back to original range
        y_pred = await this.inverse_t_y(y_pred)

        // return in JS array format
        return await y_pred.data()
    }

    async score(X, y){
        X = tf.tensor2d(X)
        y = tf.tensor1d(y)
        var y_pred = await this.predict(X)

        var score = await r2_score(y, y_pred)
        return score
    }
}

async function train_model(X, y){
    // training / validation splits
    var [X_train, X_val, y_train, y_val] = train_test_split(X, y, 0.25)

    log_progress('Training 1/1 models ...', 'models')

    // train the model
    var model = new SGDRegressor({
        'alpha': 0.0001,
        'l1_ratio': 0.15,
        'max_iter': 256,
        'eta0': 0.01,
        'batch_size': 128
    })

    await model.fit(X, y)

    // return the model
    return model
}

/**
 * Calculates how accurately the features of 
 * concept B can be predicted given the features
 * of concept A.
 * @param {Array} X Input concept features
 * @param {Array} Y Output concept features
 */
async function mapping_power(X, Y){
    var scores = []
    var N_out = Y[0].length

    // iterate over all the outputs
    for(var n_out=0; n_out<N_out; n_out++){
        var y = Y.map((v)=>v[n_out]) // select the output column

        // filter out the missing outputs
        var skip = y.map((v)=>isNaN(v))

        // filter out the missing feature values
        for(var j=0; j<X[0].length; j++){
            skip = X.map((v, i)=>skip[i]||isNaN(v[j]))
        }

        var Xf = X.filter((v, i)=>!skip[i])
        var yf = y.filter((v, i)=>!skip[i])

        // run training 
        var [X_train, X_test, y_train, y_test] = train_test_split(Xf, yf, 0.25)

        var model = await train_model(X_train, y_train)

        // score the final, best model
        var score = await model.score(X_test, y_test)
        scores.push(score)
    }

    var mean_score = scores.reduce((p, c)=>p+c)
    mean_score = mean_score / scores.length
    return mean_score
}

/**
 * Main function, used to run the computations with
 * a single file. Currently this runs the computations
 * as well as the visualization of the results into
 * the GUI of the user.
 */
async function main(results, file){
    // convert the results into dict of features

    // run 1-to-1 on all combinations
    var D = extract_concepts(results)

    var N = 0; // total size of relations
    for(var k in D){
        N++
    }
    var idx = 1

    var results = {}
    for(var from_concept in D){
        results[from_concept] = {}
        for(var to_concept in D){
            // no need to estimate concept itself
            if(from_concept === to_concept){
                continue
            }

            // select relevant data
            var X = D[from_concept]
            var Y = D[to_concept]

            // how accurate can Y be predicted with X?
            var score = await mapping_power(X, Y)

            results[from_concept][to_concept] = score
            log_progress(
                idx++ + "/" + (N*N-N) + ": " + from_concept + " ⇒ " + to_concept + ": " + score.toFixed(2), 
            'relations')
        }
    }

    // communicate results to the user ...
    log_progress(results, 'final')
}

function process_csv_file(){
    // read the csv file from the file form
    var form = document.getElementById('files')
    var file = form.files[0]
    var csv = Papa.parse(file, {
        header:true,
        complete:main,
    })
}