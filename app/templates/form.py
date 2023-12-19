{%extends "base.html" %}

{% block content %}
<p class="lead">Fill the needed data to predict if you have diabetic retinopathy!</p>
<div class = "container p-5">
    <form action="/" id="prediction-form" method="post">

        <div class="form-group col-md-6">
            <label for="inputMaculaDistance" class="form-label">Distance between optic disc and macula center</label>
            <input type="number" step="0.01" class="form-control" id="inputMaculaDistance" name="MaculaDistance" placeholder="Enter Value">
            <div id="MaculaDistancelHelp" class="form-text text-muted">Enter a float value</div>

          </div>


        <div class="mb-3">
          <label for="inputDiameter" class="form-label">Diameter of the optic disc</label>
          <input type="number" class="form-control" id="inputDiameter" name="stance = query_data.macula_opticdisc_distance">
    
          <div id="DiameterlHelp" class="form-text text-muted">Enter a float value</div>
        </div>

        <div class="mb-3">
          <label for="inputMa" class="form-label" >MA detection</label>
          <input type="number" class="form-control" class="form-control-sm" id="inputMa" name="Ma1">
          <div id="MalHelp" class="form-text text-muted">Enter a float value</div>

        </div>

        <div class="mb-3 exudates">
          <label for="inputExudate" class="form-label">Exudate1</label>
          <input type="number" class="form-control" id="inputExudate1" name="Exudate1">
             <label for="inputExudate" class="form-label">Exudate2</label>
          <input type="number" class="form-control" id="inputExudate2" name="Exudate2">
           <label for="inputExudate" class="form-label">Exudate3</label>
          <input type="number" class="form-control" id="inputExudate3" name="Exudate3">
           <label for="inputExudate" class="form-label">Exudate4</label>
          <input type="number" class="form-control" id="inputExudate31" name="Exudate31">
           <label for="inputExudate" class="form-label">Exudate5</label>
          <input type="number" class="form-control" id="inputExudate5" name="Exudate5">

          <div id="ExudatelHelp" class="form-text text-muted">Enter a float value</div>

        </div>

        <button type="submit" class="btn btn-primary" onclick="setAction('/predict_from_api/')">Submit</button>
      </form>

</div>
{% endblock %}