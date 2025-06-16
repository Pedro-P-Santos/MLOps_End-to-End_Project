import pandas as pd
from great_expectations.data_context import get_context

### PROFILING WITH GX ASSISTANT ### --- VISUALIZE IN GX UI IN DOCS
### Use python run_profiling.py to run this script


# Load your raw data
df = pd.read_csv("data/01_raw/bank-additional-full.csv", sep=";")

# Initialize GX context
context = get_context(context_root_dir="gx")

# Register in-memory pandas datasource
datasource = context.sources.add_or_update_pandas(name="profiling_source")

# Add dataframe asset
asset = datasource.add_dataframe_asset(name="profiling_asset", dataframe=df)

# Build batch request
batch_request = asset.build_batch_request()

# Run the profiling assistant
assistant = context.assistants.onboarding
result = assistant.run(batch_request=batch_request)

# Save the generated expectation suite
suite_name = "profiling_suite"
context.add_or_update_expectation_suite(
    expectation_suite=result.get_expectation_suite(expectation_suite_name=suite_name)
)

print(f"\n Profiling suite '{suite_name}' generated and saved.\n")
print("To view it, run:\n\n    great_expectations docs\n")
