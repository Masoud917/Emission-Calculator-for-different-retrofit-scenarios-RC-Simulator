import sys, os
import numpy as np
import pandas as pd
from pathlib import Path

# matplotlib imports
import matplotlib.pyplot as plt
import matplotlib

# pvlib imports
import pvlib
from pvlib.iotools import read_epw
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location as pvLocation


from .building_physics import Zone
from .radiation import Location, Window, PhotovoltaicSurface
from .heatmap import create_heatmap
# from . import supply_system
# from . import emission_system

from .exceptions import ControllerException

required_building_columns= [
    "name_of_the_building", 'situation', 'building_count', 'window_area', 'emission_windows', 'walls_area',
    'emission_insulation', 'floor_area', 'room_vol', 'total_internal_area',
    'lighting_load', 'lighting_control', 'lighting_utilisation_factor', 'lighting_maintenance_factor',
    'u_walls', 'u_windows', 'PV_area_north', 'PV_area_south', 'PV_area_east', 'PV_area_west',
    'tilt_of_PV_north', 'tilt_of_PV_south', 'tilt_of_PV_east', 'tilt_of_PV_west',
    'ach_vent', 'ach_infl', 'ventilation_efficiency', 'thermal_capacitance_per_floor_area',
    't_set_heating', 't_set_cooling', 'occupancy'
]

valid_occupancy_files_name= [
    "coolroom", "foodstore", "gym", "hospital", "hotel",
    "industrial", "lab", "library", "multi_res",
    "museum", "office", "parking", "restaurant", "retail",
    "school", "serverroom", "single_res", "swimming"
]

required_config_columns= ['key', 'value']

required_config_rows= [
    "GSHP", "ASHP", "Electric", "ds_default", "Floor_heating",
    "Radiator", "Hydronic_heat_distribution_non_residential", "PV", "Factor_of_grid",
    "Hydronic_heat_distribution_residential", "gain_per_person", "appliance_gains",
    "max_occupancy"
]


base_dir= Path.cwd()
matplotlib.style.use('ggplot')


class Demand:
    """
    calculates heating and cooling demand of all buildings
    """
    def validate_occupancy_file(self, o_file):
        """
        validate if provided occupancy file name is valid
        """
        if o_file not in valid_occupancy_files_name:
            valid_o_files= ', '.join(valid_occupancy_files_name)
            raise ControllerException(f"Provided occupancy file name is invalid, availabe options are: {valid_o_files}")

    def get_occupancy_profile(self):
        """
        Read Occupancy Profile
        """
        o_file= self.b_info.get('occupancy', "ds_default")
        self.validate_occupancy_file(o_file)
        return pd.read_excel(
            os.path.join(base_dir,'occupancy_files',f'{o_file}.xlsx'),
            header=None
        )

    def define_zone(self):
        """
        detail of building
        """
        return Zone(
            window_area=self.b_info["window_area"],
            walls_area=self.b_info["walls_area"],
            floor_area=self.b_info["floor_area"],
            room_vol=self.b_info["room_vol"],
            total_internal_area=self.b_info["total_internal_area"],
            lighting_load=self.b_info["lighting_load"],
            lighting_control=self.b_info["lighting_control"],
            lighting_utilisation_factor=self.b_info["lighting_utilisation_factor"],
            lighting_maintenance_factor=self.b_info["lighting_maintenance_factor"],
            u_walls=self.b_info["u_walls"],
            u_windows=self.b_info["u_windows"],
            ach_vent=self.b_info["ach_vent"],
            ach_infl=self.b_info["ach_infl"],
            ventilation_efficiency=self.b_info["ventilation_efficiency"],
            thermal_capacitance_per_floor_area=self.b_info["thermal_capacitance_per_floor_area"],
            t_set_heating=self.b_info["t_set_heating"],
            t_set_cooling=self.b_info["t_set_cooling"],
            max_cooling_energy_per_floor_area=-np.inf,
            max_heating_energy_per_floor_area=np.inf,
            # heating_supply_system=supply_system_dict[b_info["heating_supply_system"]],
            # cooling_supply_system=supply_system_dict[b_info["cooling_supply_system"]],
            # heating_emission_system=emission_system_dict[b_info["heating_emission_system"]],
            # cooling_emission_system=emission_system_dict[b_info["cooling_emission_system"]],
        )

    def define_window(self):
        """
        detail building's windows
        """
        return Window(
            azimuth_tilt=0,
            alititude_tilt=90,
            glass_solar_transmittance=0.7,
            glass_light_transmittance=0.8,
            area=self.b_info["window_area"]
        )

    def calculate_hourly(self, hour, zone, window):
        """
        calculates time step's heating and cooling demand of the building
        """
        # Occupancy for the time step
        max_occupancy=self.config_data.loc['max_occupancy','value']
        gain_per_person= self.config_data.loc['gain_per_person', 'value']
        appliance_gains=self.config_data.loc['appliance_gains', 'value']

        occupancy = self.occupancy_profile.loc[hour, 0] * max_occupancy
        # Gains from occupancy and appliances
        internal_gains = occupancy * gain_per_person + \
            appliance_gains * zone.floor_area
        
        # Extract the outdoor temperature in location for that hour
        t_out = self.location_data.weather_data['drybulb_C'][hour]
        Altitude, Azimuth = self.location_data.calc_sun_position(
        latitude_deg=47.480, longitude_deg=8.536, year=2015, hoy=hour)

        window.calc_solar_gains(
            sun_altitude=Altitude, sun_azimuth=Azimuth,
            normal_direct_radiation=self.location_data.weather_data['dirnorrad_Whm2'][hour],
            horizontal_diffuse_radiation=self.location_data.weather_data['difhorrad_Whm2'][hour]
        )
        window.calc_illuminance(
            sun_altitude=Altitude, sun_azimuth=Azimuth,
            normal_direct_illuminance=self.location_data.weather_data['dirnorillum_lux'][hour],
            horizontal_diffuse_illuminance=self.location_data.weather_data['difhorillum_lux'][hour]
        )

        zone.solve_energy(internal_gains=internal_gains,
                            solar_gains=window.solar_gains,
                            t_out=t_out,
                            t_m_prev=self.t_m_prev
        )
        zone.solve_lighting(
            illuminance=window.transmitted_illuminance, occupancy=occupancy
        )
        
        # Set the previous temperature for the next time step
        self.t_m_prev = zone.t_m_next
        return zone, t_out, window

    def _get_annual_result(self, zone, window):
        """
        calculates heating and cooling demand of the building in all hours of a year
        """
        annual_results_dict= {
            'HeatingDemand':[],
            # 'HeatingEnergy':[],
            'CoolingDemand':[],
            # 'CoolingEnergy':[],
            'ElectricityOut':[],
            'IndoorAir': [],
            'OutsideTemp':  [],
            'SolarGains': [],
            'COP': [],
        }
        # Loop through all 8760 hours of the year
        for hour in range(8760):
            zone, t_out, window=self.calculate_hourly(hour, zone, window)
            annual_results_dict['HeatingDemand'].append(zone.heating_demand)
            # annual_results_dict['HeatingEnergy'].append(zone.heating_energy)
            annual_results_dict['CoolingDemand'].append(zone.cooling_demand)
            # annual_results_dict['CoolingEnergy'].append(zone.cooling_energy)
            annual_results_dict['ElectricityOut'].append(zone.electricity_out)
            annual_results_dict['IndoorAir'].append( zone.t_air)
            annual_results_dict['OutsideTemp'].append(t_out)
            annual_results_dict['SolarGains'].append(window.solar_gains)
            annual_results_dict['COP'].append(zone.cop)

        return annual_results_dict

    def get_annual_result(self):
        self.occupancy_profile= self.get_occupancy_profile()
        zone= self.define_zone()
        window= self.define_window()

        return self._get_annual_result(zone, window)

    def get_annual_demand(self):
        notb= self.b_info['name_of_the_building']
        situation= self.b_info['situation']

        try:
            annual_results_dict= self.get_annual_result()
        except ControllerException as e:
            with open(self.p_dir/ 'error/errors.txt', 'at') as f:
                f.write(f'error in index {self.b_index} config file: {str(e)}\n')
        
        # get summation of all columns(annual demand)
        annual_df = pd.DataFrame(annual_results_dict)
        sum_row = annual_df.sum(axis=0)
        annual_df.loc['summation']= sum_row

        result_path= self.get_hd_file_dir(notb, situation)
        annual_df.to_excel(result_path)
        heating_demand= annual_df.loc['summation', 'HeatingDemand']

        return heating_demand


class Emission:
    def calculate_heating_energy(self, heating_demand):
        return {
            "pellet": heating_demand / 0.7,
            "boiler_old": heating_demand / 0.63,
            "boiler_med": heating_demand / 0.82,
            "boiler_new": heating_demand / 0.98,
            "electric": heating_demand / 1,
            "ASHP": heating_demand / 2.3,
            "GSHP": heating_demand / 3.67,
        }
    
    def calculate_energy_emission(self, heating_demand):
        all_heating_energy= self.calculate_heating_energy(heating_demand)
        fog= self.config_data.loc['Factor_of_grid','value']
        for heating_energy in all_heating_energy:
            energy_emission= all_heating_energy[heating_energy] / 1000 * fog
            all_heating_energy[heating_energy]= energy_emission

        return all_heating_energy

    def calculate_GSHP_emission(self):
        return self.config_data.loc['GSHP', 'value'] * self.b_info['size_GSHP']

    def calculate_ASHP_emission(self):
        return self.config_data.loc['ASHP', 'value'] * self.b_info['size_ASHP']
            
    def calculate_electric_emission(self):
        return self.b_info['size_electric'] * self.config_data.loc['Electric', 'value']

    def calculate_insulation_emission(self):
        return self.b_info['emission_insulation'] * self.b_info['walls_area']

    def calculate_windows_emission(self):
        return self.b_info['emission_windows'] * self.b_info['window_area']

    def get_annual_emission(self, heating_demand, pv_production):
        floor_area= self.b_info['floor_area']
        energy_emission= self.calculate_energy_emission(heating_demand)
        emission_insulation_wall= self.calculate_insulation_emission()
        emssion_windows= self.calculate_windows_emission()

        ds_name= self.b_info['distribution_system']
        ds_emission= self.config_data.loc[ds_name, 'value'] * floor_area

        b_emission= emission_insulation_wall + emssion_windows + ds_emission
        
        emission_electric=self.calculate_electric_emission()
        emission_GSHP=self.calculate_GSHP_emission()
        emission_ASHP=self.calculate_ASHP_emission()

        result= {
            "GSHP+PVs": ((energy_emission['GSHP']+b_emission+emission_GSHP) - pv_production)/ floor_area * 60,
            "GSHP": (energy_emission['GSHP']+b_emission+emission_GSHP)/ floor_area * 60,
            "ASHP+PVs": ((energy_emission['ASHP']+b_emission+emission_ASHP) - pv_production)/ floor_area * 60,
            "ASHP": (energy_emission['ASHP']+b_emission+emission_ASHP)/ floor_area * 60,
            "Electrical heating+PVs": ((energy_emission['electric']+b_emission+emission_electric) - pv_production)/ floor_area * 60,
            "Electrical heating": (energy_emission['electric']+b_emission+emission_electric)/ floor_area * 60,
            "Pellet+PVs": ((energy_emission['pellet']+b_emission) - pv_production)/ floor_area * 60,
            "Pellet": (energy_emission['pellet']+b_emission)/ floor_area * 60,
            "Old boiler+PVs": ((energy_emission['boiler_old']+b_emission) - pv_production)/ floor_area * 60,
            "Old boiler": (energy_emission['boiler_old']+b_emission)/ floor_area * 60,
            "Med boiler+PVs": ((energy_emission['boiler_med']+b_emission) - pv_production)/ floor_area * 60,
            "Med boiler": (energy_emission['boiler_med']+b_emission)/ floor_area * 60,
            "New boiler+PVs": ((energy_emission['boiler_new']+b_emission) - pv_production)/ floor_area * 60,
            "New boiler": (energy_emission['boiler_new']+b_emission)/ floor_area * 60,
        }

        return result
    

class PV:
    def read_weather_data(self):
        self.latitude = self.epw_meta['latitude']
        self.longitude = self.epw_meta['longitude']
        self.altitude = self.epw_meta['altitude']
        self.timezone = self.epw_meta['TZ']
        weather = pvlib.iotools.get_pvgis_tmy(self.latitude, self.longitude)[0]
        weather.index.name = "utc_time"
        self.weather = weather

    def load_system(self, direction):
        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
        module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
        inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        location = pvLocation(
            self.latitude,
            self.longitude,
            name="utc_time",
            altitude=self.altitude,
            tz=self.timezone,
        )
        mount = FixedMount(
            surface_tilt= self.pv_info[direction]["tilt"],
            surface_azimuth= self.pv_info[direction]["surface_azimuth"]
        )
        array = Array(
            mount=mount,
            module_parameters=module,
            temperature_model_parameters=temperature_model_parameters,
        )
        self.system = PVSystem(arrays=[array], inverter_parameters=inverter)
        self.mc = ModelChain(self.system, location)

    def run_model_chain(self):
        self.mc.run_model(self.weather)
        return self.mc.results.ac.sum()

    def get_init_data(self):
        self.read_weather_data()
        self.pv_info= {
            "north":{
                "surface_azimuth":0,
                "area": self.b_info.get("PV_area_north", 0),
                "tilt": self.b_info.get("tilt_of_PV_north", 0)
            },
            "east":{
                "surface_azimuth":90,
                "area": self.b_info.get("PV_area_east", 0),
                "tilt": self.b_info.get("tilt_of_PV_east", 0)
            },
            "south":{
                "surface_azimuth":180,
                "area": self.b_info.get("tilt_of_PV_south", 0),
                "tilt": self.b_info.get("PV_area_south", 0)
            },
            "west":{
                "surface_azimuth":270,
                "area": self.b_info.get("PV_area_west", 0),
                "tilt": self.b_info.get("tilt_of_PV_west", 0)
            },
        }

    def get_energy_of_each_side(self, direction):
        self.load_system(direction)
        pvlib_result= self.run_model_chain()
        return pvlib_result / 1000 * self.pv_info[direction]["area"]
    
    def get_emission_of_each_side(self, direction):
        return self.config_data.loc['PV','value'] * self.pv_info[direction]["area"]

    def get_annual_pv_result(self):
        self.get_init_data()

        pv_energy= 0
        pv_emission= 0

        for direction in ['north', 'east', 'south', 'west']:
            pv_energy+= self.get_energy_of_each_side(direction)
            pv_emission+= self.get_emission_of_each_side(direction)

        pv_energy= pv_energy * self.config_data.loc['Factor_of_grid','value']

        return pv_energy, pv_emission
        

class AnnualHeating(Demand, Emission, PV):
    def __init__(self, project_dir, building_info_file, config_file, epw_file):
        self.p_dir= project_dir
        self.config_data= self.get_config_data(config_file)
        self.buildings_data= self.get_building_data(building_info_file)
        self.get_epw_data(epw_file)

        # Starting temperature of the builidng
        self.t_m_prev = 20
    
    def validate_config_df(self, c_df):
        c_error_message= None
        r_error_message= None
        missing_rows = []
        if "value" not in c_df.columns:
            c_error_message= f"value column is missing in config excel file, make sure you have provided it and has the same name"

        for row in required_config_rows:
            if row not in c_df.index.to_list():
                missing_rows.append(row)

        if missing_rows:
            missing_rows_msg= ', '.join(missing_rows)
            r_error_message= f"These rows are missing in config excel file, make sure you have provided them and have the same name: {missing_rows_msg}"
        
        if c_error_message and r_error_message:
            raise ControllerException(c_error_message + "\n" + r_error_message)
        elif c_error_message:
            raise ControllerException(c_error_message)
        elif r_error_message:
            raise ControllerException(r_error_message)
        else:
            return
        
    def validate_building_df(self, df):
        missing_columns = []
        for column in required_building_columns:
            if column not in df.columns:
                missing_columns.append(column)
        
        if missing_columns:
            missing_columns_msg= ', '.join(missing_columns)

            raise ControllerException(
                f"These columns are missing in building excel file, make sure you have provided them and have the same name: {missing_columns_msg}")

    def get_config_data(self, config_file):
        try:
            c_df= pd.read_excel(config_file, index_col="key")
        except:
            raise ControllerException("key column is missing in config excel file, make sure you have provided it and has the same name")
        
        self.validate_config_df(c_df)

        return c_df

    def get_epw_data(self, epw_file_path):
        self.location_data= Location(epwfile_path=os.path.join(epw_file_path))
        self.epw_meta = read_epw(epw_file_path)[1]

    def get_building_data(self, building_info_file):
        df= pd.read_excel(building_info_file, skiprows=[1,])
        self.validate_building_df(df)
        all_unique_buldings= list(set(df['name_of_the_building'].tolist()))
        for building_name in all_unique_buldings:
            rows_with_desired_value = df.query(f"name_of_the_building == '{building_name}'")
            for index, building_info in rows_with_desired_value.iterrows():
                yield index, building_info
    
    def sanitize_string(self, string):
        invalid_chars = ['/','\\', ':', '*', '?', '"', '|', '<', '>']
        for invalid_char in invalid_chars:
            string = string.replace(invalid_char, '_')
        return string

    def get_result_dir(self, notb):
        notb= self.sanitize_string(notb)

        file_directory= self.p_dir/f"result/{notb}"
        if not file_directory.exists():
            file_directory.mkdir(parents= True)

        return file_directory

    def get_hd_file_dir(self, notb, situation):
        """
        heating demand file directory
        """
        situation= self.sanitize_string(situation)
        file_directory= self.get_result_dir(notb)

        file= file_directory / f"{situation}.xlsx"
        duplicate_file_number= 0
        while file.exists():
            duplicate_file_number+=1
            file= file_directory / + f"{situation}_{duplicate_file_number}.xlsx"
        
        return file

    def create_retrofit_table_files(self, retrofit_table):
        for notb, b_info in retrofit_table.items():
            df= pd.DataFrame(b_info).T
            result_path= self.get_result_dir(notb)
            df.to_excel(result_path / "Retrofit_Table.xlsx")
            create_heatmap(df, notb, result_path / "Retrofit_Table.png")

    def create_pv_production_table_files(self, pv_production_table):
        for notb, b_info in pv_production_table.items():
            df= pd.DataFrame(b_info).T
            result_path= self.get_result_dir(notb)
            df.to_excel(result_path / "PV_Production.xlsx")

    def create_district_summation_table_files(self, retrofit_table):
        common_situations_sum = {}
        for notb_key, situations in retrofit_table.items():
            for situation_key, situation_values in situations.items():
                # Check if the situation is already in the common_situations_sum dictionary
                if situation_key in common_situations_sum:
                    # Iterate through the values of the current situation and add them to the sum
                    for value_key, value in situation_values.items():
                        common_situations_sum[situation_key][value_key] = common_situations_sum[situation_key].get(value_key, 0) + value
                else:
                    # If the situation is not in the common_situations_sum dictionary, add it with its values
                    common_situations_sum[situation_key] = situation_values.copy()

        df= pd.DataFrame(common_situations_sum).T
        result_path= self.get_result_dir("District_Summation")
        df.to_excel(result_path / "District_Summation.xlsx")
        create_heatmap(df, "District Summation", result_path / "District_Summation.png")

    def process_building(self, b_data):
        self.b_index = b_data[0]
        self.b_info = b_data[1]

        heating_demand = self.get_annual_demand()

        pv_energy, pv_emission = self.get_annual_pv_result()
        pv_production = pv_energy - pv_emission

        annual_emission = self.get_annual_emission(heating_demand, pv_production)

        return pv_energy, annual_emission

    def execute(self):
        retrofit_table = {}
        district_summation_table= {}
        pv_production_table = {}

        for b_data in self.buildings_data:
            pv_energy, annual_emission = self.process_building(b_data)

            notb = self.b_info['name_of_the_building']
            situation = self.b_info['situation']

            pv_production_table.setdefault(notb, {})
            pv_production_table[notb][situation]= [pv_energy]

            retrofit_table.setdefault(notb, {})
            retrofit_table[notb][situation]= annual_emission

            for key, value in annual_emission.items():
                district_summation_table.setdefault(notb, {})
                district_summation_table[notb].setdefault(situation, {})
                district_summation_table[notb][situation][key]= value * self.b_info['building_count']

        self.create_retrofit_table_files(retrofit_table)
        self.create_district_summation_table_files(district_summation_table)
        self.create_pv_production_table_files(pv_production_table)

# directory of the required files
project_dir=''
building_info_file=''
config_file=''
epw_file=''

AnnualHeating(project_dir, building_info_file, config_file, epw_file).execute()