#!/usr/bin/env julia
# Script to retrain all four metalearner models
# Author: Claude Code
# Date: 2026-02-03

println("="^80)
println("Starting metalearner training pipeline")
println("="^80)

# Activate the environment
using Pkg
Pkg.activate(".")

# Load the package
println("\nLoading BayesInteractomics package...")
using BayesInteractomics

# Define paths
dtrain_path = "encodings/train_data.h5"
dval_path   = "encodings/val_data.h5"
dtest_path  = "encodings/test_data.h5"
model_path  = "encodings/model-927-0.4975538112736281.jld2"


# Verify files exist
println("\nVerifying training data files...")
for (name, path) in [
    ("Training data", dtrain_path),
    ("Validation data", dval_path),
    ("Test data", dtest_path),
    ("DNN model", model_path)
]
    if isfile(path)
        println("  ✓ $name: $path")
    else
        error("  ✗ Missing $name: $path")
    end
end

# Create metalearners directory if it doesn't exist
if !isdir("metalearners")
    mkdir("metalearners")
    println("\nCreated metalearners/ directory")
else
    println("\nUsing existing metalearners/ directory")
end

println("\n" * "="^80)
println("MODEL 1/4: HistGradientBoostingClassifier")
println("="^80)
try
    mach1 = BayesInteractomics.fit_HistGradientBoostingClassifier(dtrain_path, dval_path, dtest_path, model_path)
    println("\n✓ HistGradientBoostingClassifier trained successfully!")
    println("  Saved to: metalearners/HistGradienBossting_tune.jld2")
catch e
    println("\n✗ Error training HistGradientBoostingClassifier:")
    println(e)
    rethrow(e)
end

println("\n" * "="^80)
println("MODEL 2/4: LogisticClassifier")
println("="^80)
try
    mach2 = BayesInteractomics.fit_LogisticClassifier(dtrain_path, dval_path, dtest_path, model_path)
    println("\n✓ LogisticClassifier trained successfully!")
    println("  Saved to: metalearners/LogisticClassifier_tune.jld2")
catch e
    println("\n✗ Error training LogisticClassifier:")
    println(e)
    rethrow(e)
end

println("\n" * "="^80)
println("MODEL 3/4: GaussianNBClassifier")
println("="^80)
try
    mach3 = BayesInteractomics.fit_GaussianNBClassifier(dtrain_path, dval_path, dtest_path, model_path)
    println("\n✓ GaussianNBClassifier trained successfully!")
    println("  Saved to: metalearners/GaussianNBC_tune.jld2")
catch e
    println("\n✗ Error training GaussianNBClassifier:")
    println(e)
    rethrow(e)
end

println("\n" * "="^80)
println("MODEL 4/4: Ensemble")
println("="^80)
try
    mach4 = BayesInteractomics.fit_Ensemble(dtrain_path, dval_path, dtest_path, model_path)
    println("\n✓ Ensemble trained successfully!")
    println("  Saved to: metalearners/ensemble.jld2")
catch e
    println("\n✗ Error training Ensemble:")
    println(e)
    rethrow(e)
end

println("\n" * "="^80)
println("TRAINING COMPLETE")
println("="^80)

# Verify all models were saved
println("\nVerifying saved models:")
saved_models = [
    "metalearners/HistGradienBossting_tune.jld2",
    "metalearners/LogisticClassifier_tune.jld2",
    "metalearners/GaussianNBC_tune.jld2",
    "metalearners/ensemble.jld2"
]

all_saved = true
for model_file in saved_models
    if isfile(model_file)
        filesize_mb = filesize(model_file) / (1024^2)
        println("  ✓ $model_file ($(round(filesize_mb, digits=2)) MB)")
    else
        println("  ✗ Missing: $model_file")
        all_saved = false
    end
end

if all_saved
    println("\n🎉 All four metalearner models trained and saved successfully!")
else
    println("\n⚠ Warning: Some models may not have been saved properly")
end

println("\nTraining pipeline complete.")
println("="^80)
