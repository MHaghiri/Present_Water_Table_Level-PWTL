# Present_Water_Table_Level(PWTL)

This repository use the [Water Table Model (WTM)](https://github.com/KCallaghan/WTM) to simulate the present-day water table levels across various regions. The simulations aim to provide insights into groundwater behavior using robust modeling techniques.

## Repository Structure

The repository is structured into the following folders:

- **`codes/`**: Contains all Python scripts for:
  - Preparing input data.
  - Generating individual layers.
  - Extracting and processing blocks of data at yearly, monthly, and seasonal scales.
  - Performing Bias Correction and Spatial Downscaling (BCSD).
  - Calculating variables such as Vapor Pressure Deficit (VPD).
  - Converting units for datasets (e.g., mm to m).

### Example of Codes Included:
- `all_1year_block_extraction.py`: Extracts one-year blocks from datasets.
- `all_monthly_block_extraction.py`: Extracts monthly blocks from datasets.
- `Calculate_VPD.py`: Script to calculate Vapor Pressure Deficit (VPD).
- `convert_mm_to_m.py`: Converts data units from mm to m for standardization.

## Simulations and Outputs

The simulations conducted in this repository aim to:
- Reconstruct present-day water table levels using the WTM.
- Integrate high-resolution datasets for better accuracy and representation.
- Provide layered outputs for further analysis.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Present_Water_Table_Level.git



2. Navigate to the codes directory:

   
   cd Present_Water_Table_Level-(PWTL)/codes



3. Execute scripts for specific data preparation or simulations. For example:

   python Calculate_VPD.py

 

4. Review outputs stored in their respective directories.



## License
This repository is licensed under MIT Licence.



## Contributions

This repository is developed in collaboration with Professor Callaghan, whose expertise and guidance have been instrumental in using the Water Table Model (WTM) to simulate present-day water table levels. Contributions to this project include the development of scripts for data preparation, bias correction, spatial downscaling, and the integration of high-resolution datasets.

If you would like to contribute to this repository, feel free to submit issues or create pull requests. Suggestions and improvements are always welcome.
