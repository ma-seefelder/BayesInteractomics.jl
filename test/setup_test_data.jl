using DataFrames
using CSV
using XLSX

# Create dummy_data directory if it doesn't exist
if !isdir("test/dummy_data")
    mkdir("test/dummy_data")
end

# Generate meaningful data
# Columns:
# 1: ID/Name
# 2-4: Control Protocol 1
# 5-7: Control Protocol 2 (Overlap with 6,7 in P3?)
# 8-10: Sample Protocol 1
# 11-13: Sample Protocol 2
# 14-15: Sample Protocol 3
# 162-165: Additional columns referenced in src/data_loading.jl defaults but likely not used in this specific test

# Let's just make 20 columns of random data
df = DataFrame(
    Col1=["P1", "P2", "P3"],
    Col2=rand(3), Col3=rand(3), Col4=rand(3),
    Col5=rand(3), Col6=rand(3), Col7=rand(3),
    Col8=rand(3), Col9=rand(3), Col10=rand(3),
    Col11=rand(3), Col12=rand(3), Col13=rand(3),
    Col14=rand(3), Col15=rand(3), Col16=rand(3),
    Col17=rand(3), Col18=rand(3), Col19=rand(3),
    Col20=rand(3)
)

# Write CSV
CSV.write("test/dummy_data/dummy_data.csv", df)

# Write XLSX
XLSX.writetable("test/dummy_data/dummy_data.xlsx", "Sheet1" => df)

println("Dummy data created successfully.")
