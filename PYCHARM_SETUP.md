# Run in PyCharm

## Fastest setup on Windows

1. Extract the ZIP.
2. Open the extracted `ml_recommender_system` folder in PyCharm.
3. Open PyCharm's Terminal panel.
4. Run:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\setup_pycharm.ps1
   ```

5. Go to **File > Settings > Project > Python Interpreter**.
6. Select **Add Interpreter > Add Local Interpreter > Existing**.
7. Choose:

   ```text
   <project>\.venv\Scripts\python.exe
   ```

8. Select the shared **Web UI - Streamlit** run configuration and click Run.
9. PyCharm will start the local web server and open the dashboard in your browser.

Expected result:

- the model is loaded or trained automatically
- the browser displays the interactive career dashboard
- changing the student preset updates the recommendations
- the mastery and evidence editor supports custom profiles
- recommendation JSON can be downloaded from the dashboard

## PyCharm run configurations

- **Web UI - Streamlit** - launches the interactive dashboard
- **Demo Recommendation** - runs `main.py`
- **Retrain and Demo** - rebuilds the model and runs the example
- **Evaluate Model** - runs the benchmark and perturbation evaluation
- **Run Unit Tests** - runs all ten model and UI tests

## Manual interpreter setup

If PowerShell script execution is unavailable:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Then point PyCharm to `.venv\Scripts\python.exe`.

## Run Streamlit manually

From the PyCharm Terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The dashboard opens at `http://localhost:8501`.

## Use another student

Copy `examples/student_profile.json`, edit the scores and evidence counts, then set this in the `main.py` run configuration parameters:

```text
--profile path\to\your_profile.json --top-k 3
```
