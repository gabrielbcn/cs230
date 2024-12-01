"""
The Scene class

by
Gabriel Mesquida Masana
gabmm@stanford.edu

Written for Python 3.11
"""

import copy
from flight import Flight as Fl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Self, Iterator


class Scene:
    """
    The Scene class
    ================
    Version: 1.45
    Last update: 01/12/24
    -----------------------
    Gabriel Mesquida Masana
    gabmm@stanford.edu

    A Scene object contains:

    * flights: a list of Flight objects
    * notes: a string of the manipulations done to the Scene
    * match_transform: parameters used for match standardisation

    Methods:
        * __init__
            - from_file
            - from_scene
            - from_flights
        * __repr__

    for filtering scenes or sets of flights
        * arrivals
        * departures
        * overflights
        * aircraft
        * mintracks
        * mintime
        * split_c
        * remove
        * split_train_dev_test

    for filtering tracks within flights
    (always makes deepcopy of modified flights)
        * descent
        * climb (refreshes alt_max)
        * higher_than
        * downsample

    for projecting fixes
        * project
        * match_project
        * match_convert_geo_to_enu
        * match_standardise_geo
        * match_standardise_enu
        * match_filter_duration

    for querying scene
        * alt_max (option to refresh)
        * alt_min
        * c_max_max (first flight, second scene)
        * c_max_min
        * query_attribute (can produce pie chart)

    for rendering
        * plot
        * plot_elevation_profile
        * plot_elevation_histogram
        * plot_c (histogram)
        * plot_alt_max (density)


    """

    #
    # Constants
    #
    MINCONTTIME = 600  # 600 seconds = 10 min
    MINMATCHTIME = 900  # 1200 seconds = 15 min

    def __init__(
        self,
        from_file: str = None,
        from_scene: Self = None,
        from_flights: list[Fl] = None,
        notes: str = None,
        match_transform: dict = None,
    ):
        """
        Creating objects:
            - from_file
            - from_scene
            - from_flights
        """

        # Declaring all fields in __init__
        self.flights = None
        self.notes = notes
        self.match_transform = match_transform

        # If loading
        if from_file:
            Fl.load_dataframe(file_name=from_file)
            self.flights = Fl.create_flights()
            self.notes = f"loaded from {from_file}"

        # From scene
        elif from_scene:
            self.flights = copy.copy(from_scene.flights)
            self.notes = f"copy of object ({from_scene.notes})"

        # From flights
        elif from_flights:
            self.flights = copy.copy(from_flights)

    def match_filter_duration(self, seconds: int = MINMATCHTIME) -> Self:
        """
        Helper to only include the flights that are long enough
        """
        self.flights = [
            flight
            for flight in self.flights
            if len(flight.match) >= seconds / flight.additional["time_gap"]
        ]
        self.notes += f", match longer than {seconds} seconds"
        if len(self.flights) == 0:
            print("Error: empty scene!")
        return self

    def arrivals(self, arriving_from: str = None) -> Self:
        """
        Filters for arrivals
        """
        self.flights = list(
            filter(
                lambda flight: (
                    flight.ades == "WSSS"
                    if not arriving_from
                    else flight.ades == "WSSS" and flight.adep == arriving_from
                ),
                self.flights,
            )
        )
        self.notes += (
            ", arriving to WSSS"
            if not arriving_from
            else f", arriving to WSSS from {arriving_from}"
        )
        return self

    def departures(
        self, departing_to: str = None, clean_altitudes: bool = False
    ) -> Self:
        """
        Filters for departures
        """
        self.flights = list(
            filter(
                lambda flight: (
                    flight.adep == "WSSS"
                    if not departing_to
                    else flight.adep == "WSSS" and flight.ades == departing_to
                ),
                self.flights,
            )
        )
        self.notes += (
            ", departing WSSS"
            if not departing_to
            else f", departing WSSS to {departing_to}"
        )
        if clean_altitudes == True:
            filtered = list(
                filter(
                    lambda flight: flight.tracks.iloc[0].Altitude < 500,
                    self.flights,
                )
            )
            if len(filtered) != len(self.flights):
                print(f"Removed {len(self.flights)-len(filtered)} flights")
                self.flights = filtered
                self.notes += (
                    f", removed {len(self.flights)-len(filtered)} flights"
                )
        return self

    def overflights(self) -> Self:
        """
        Filters for overflights
        """
        self.flights = list(
            filter(
                lambda flight: flight.adep != "WSSS" and flight.ades != "WSSS",
                self.flights,
            )
        )
        self.notes += ", overflights"
        return self

    def mintracks(self, tracks: int = 75) -> Self:
        """
        Filters for minimum number of tracks
        """
        self.flights = list(
            filter(
                lambda flight: (len(flight.tracks) > tracks),
                self.flights,
            )
        )
        self.notes += f", with min {tracks} tracks"
        if len(self.flights) == 0:
            print("Error: empty scene!")
        return self

    def mintime(self, seconds: int = MINCONTTIME) -> Self:
        """
        Helper to only include the flights that are long enough
        """
        self.flights = [
            flight
            for flight in self.flights
            if len(flight.tracks) >= seconds / flight.additional["time_gap"]
        ]
        self.notes += f", tracks longer than {seconds} seconds"
        if len(self.flights) == 0:
            print("Error: empty scene!")
        return self

    def aircraft(self, aircraft: str = "A320") -> Self:
        """
        Filters for overflights
        """
        self.flights = list(
            filter(
                lambda flight: flight.aircraft == aircraft,
                self.flights,
            )
        )
        self.notes += f", aircraft is {aircraft}"
        if len(self.flights) == 0:
            print("Error: empty scene!")
        return self

    def descent(self) -> Self:
        """
        Filters the tracks for descent segment only
        """
        cut_flights = copy.deepcopy(self.flights)
        for f_ in cut_flights:
            # 20 seconds same level
            f_.tracks["dA"] = f_.tracks["Altitude"].diff(-6)

            # block that is going down
            flat = f_.tracks[f_.tracks.dA > 40]
            if len(flat) > 0:
                f_.tracks = f_.tracks.iloc[flat.iloc[0].name + 1 :]
            f_.tracks.drop(columns=["dA"], inplace=True)
            f_.numtracks = len(f_.tracks)
        self.flights = cut_flights
        self.notes += ", descent only"
        return self

    def downsample(self, factor: int) -> Self:
        """
        Thinning tracks by a factor
        Tracks will be every time_gap * factor seconds
        """
        thin_flights = copy.deepcopy(self.flights)
        for f_ in thin_flights:
            f_.downsample(factor=factor)
            if not "alt_max" in f_.additional:
                f_.additional["alt_max"] = f_.tracks["Altitude"].max()
            f_.match = None
        self.flights = thin_flights
        self.notes += f", thinned to every {f_.additional['time_gap']}s"
        return self

    def climb(self) -> Self:
        """
        Filters the tracks for climb segment only
        """
        cut_flights = copy.deepcopy(self.flights)
        for f_ in cut_flights:
            # 20 seconds same level
            f_.tracks["dA"] = f_.tracks["Altitude"].diff(-6)
            # block that is flat or going down
            flat = f_.tracks[f_.tracks.dA >= 0]
            if len(flat) > 0:
                f_.tracks = f_.tracks.iloc[0 : flat.iloc[0].name + 10]
            # Clean
            f_.tracks.drop(columns=["dA"], inplace=True)
            f_.additional["alt_max"] = f_.tracks["Altitude"].max()
            f_.numtracks = len(f_.tracks)
        self.flights = cut_flights
        self.notes += ", climb only"
        return self

    def higher_than(self, altitude: float) -> Self:
        """
        Filters the tracks and removes segment lower than altitude threshold
        """
        cut_flights = copy.deepcopy(self.flights)
        for f_ in cut_flights:
            f_.tracks = f_.tracks[f_.tracks.Altitude >= altitude]
            f_.numtracks = len(f_.tracks)
        self.flights = cut_flights
        self.notes += f", higher than {altitude}"
        return self

    def alt_max(self, refresh: bool = False) -> float:
        """
        Returns max altitude in scene
        """
        if refresh:
            for f_ in self.flights:
                f_.additional["alt_max"] = f_.tracks["Altitude"].max()
        return max(
            map(lambda flight: flight.additional["alt_max"], self.flights)
        )

    def plot_alt_max(self, clip: tuple[float, float] = None) -> None:
        maxs = [flight.additional["alt_max"] for flight in self.flights]
        sns.kdeplot(maxs, bw=0.01, fill=True, clip=clip)
        plt.title(f"Alt Max for {len(self.flights)} flights")
        plt.yticks(None)
        plt.show()

    def alt_min(self) -> float:
        """
        Returns min altitude in scene
        """
        return min(
            map(
                lambda flight: min(flight.additional["alt_max"].to_list()),
                self.flights,
            )
        )

    def c_max_min(self) -> float:
        """
        Returns min c_max value in scene
        """
        return min(
            map(lambda flight: flight.additional["c_max"], self.flights)
        )

    def c_max_max(self) -> float:
        """
        Returns max c_max value in scene
        """
        return max(
            map(lambda flight: flight.additional["c_max"], self.flights)
        )

    def plot_c(self) -> None:
        """
        Plots c_max value density distribution from scene
        """
        cs = [flight.additional["c_max"] for flight in self.flights]
        clip = (self.c_max_min(), self.c_max_max())
        _, (ax1, ax2) = plt.subplots(2)
        sns.kdeplot(cs, bw=0.02, fill=True, ax=ax1, clip=clip)
        sns.boxplot(x=cs, ax=ax2)
        plt.suptitle(f"c_max distribution for {len(self.flights)} flights")
        plt.show()

    def split_c(self, c_threshold: float = 2) -> tuple[Self, Self]:
        """
        Split one scene into two based on c_max threshold value
        """
        flights_c_low = [
            flight
            for flight in self.flights
            if float(flight.additional["c_max"]) <= c_threshold
        ]
        flights_c_high = [
            flight
            for flight in self.flights
            if float(flight.additional["c_max"]) > c_threshold
        ]
        return (
            Scene(
                from_flights=flights_c_low,
                notes=f"{self.notes}, c<={c_threshold}",
                match_transform=self.match_transform,
            ),
            Scene(
                from_flights=flights_c_high,
                notes=f"{self.notes}, c>{c_threshold}",
                match_transform=self.match_transform,
            ),
        )

    def split_train_dev_test(
        self, trainpct: float = 80, devpct: float = 10
    ) -> tuple[Self, Self, Self]:
        """
        Returns three different scenes for train, dev, and test
        """
        testpct = 100 - trainpct - devpct
        place = np.random.choice(
            3,
            len(self.flights),
            p=[trainpct / 100, devpct / 100, testpct / 100],
        )
        train, dev, test = ([], [], [])

        for n_, f_ in enumerate(self.flights):
            match place[n_]:
                case 0:
                    train.append(f_)
                case 1:
                    dev.append(f_)
                case 2:
                    test.append(f_)

        return (
            Scene(
                from_flights=train,
                notes=f"{self.notes}, {trainpct}% for train",
                match_transform=self.match_transform,
            ),
            Scene(
                from_flights=dev,
                notes=f"{self.notes}, {devpct}% for dev",
                match_transform=self.match_transform,
            ),
            Scene(
                from_flights=test,
                notes=f"{self.notes}, {testpct}% for test",
                match_transform=self.match_transform,
            ),
        )

    def remove(self, id: str) -> Self:
        """
        Removes one or more flights from scene
        """
        new_list = []
        if isinstance(id, str):
            id = [id]

        for f_ in self.flights:
            if f_.flight_id in id:
                print(f"Removed {f_.flight_id}")

            else:
                new_list.append(f_)
        self.flights = new_list
        self.notes = f"{self.notes}, removed {len(id)} flights"
        return self

    #
    # More specific queries
    #
    def query_attribute(
        self,
        attribute: str = "aircraft",
        n: int = 10,
        pie: bool = False,
    ) -> None | pd.Series:
        """
        Can retrieve values of an attribute or create pie chart
        Attributes allowed are ["Aircraft", "ADES", "ADEP", "Callsign"]
        """

        if attribute.lower() not in ["aircraft", "ades", "adep", "callsign"]:
            raise TypeError("This attribute is not allowed")
        airframes = pd.Series(
            [getattr(f_, attribute.lower()) for f_ in self.flights]
        )
        if pie:
            values = [
                str(val_[0]) + " (" + str(val_[1]) + ")"
                for val_ in zip(
                    airframes.value_counts().head(n).index,
                    (airframes.value_counts().head(n)),
                )
            ]
            airframes.value_counts().head(n).plot.pie(
                labels=values, autopct="%1.1f%%"
            )
            plt.title(f"{attribute} (main {n})")
        else:
            return airframes.value_counts().head(n)

    #
    # Wrappers
    #
    def project(self) -> Self:
        """
        Creates fixes projection in a linear way
        """
        [flight.project() for flight in self.flights]
        self.notes += f", created projection"
        return self

    def match_project(self) -> Self:
        """
        Creates fixes projections and matches them with tracks
        """
        [flight.match_projection() for flight in self.flights]
        self.mintime()
        self.notes += (
            f", matched projections, {len(self.flights)} flights left"
        )
        return self

    def match_convert_geo_to_enu(self) -> Self:
        """
        Converts fixes projections and tracks from geodesic to ENU
        """
        [flight.match_geo_to_enu() for flight in self.flights]
        self.notes += f", converted Geodesic to ENU"
        return self

    def match_standardise_geo(self) -> Self:
        """
        Normalises geodetic projections and tracks
        """
        # Scan all training set
        lat_max = max([flight.match["Trk_Lat"].max() for flight in self])
        lat_min = min([flight.match["Trk_Lat"].min() for flight in self])
        lon_max = max([flight.match["Trk_Lon"].max() for flight in self])
        lon_min = min([flight.match["Trk_Lon"].min() for flight in self])
        alt_max = max([flight.match["Trk_Alt"].max() for flight in self])
        alt_min = min([flight.match["Trk_Alt"].min() for flight in self])
        steps_max = max([len(flight.match) for flight in self])

        # Store the dictionary of the transformation
        transform = {
            "lat_max": float(lat_max),
            "lat_min": float(lat_min),
            "lon_max": float(lon_max),
            "lon_min": float(lon_min),
            "alt_max": float(alt_max),
            "alt_min": float(alt_min),
            "steps_max": steps_max,
        }
        for flight in self.flights:
            flight.match_geo_transform(transform)

        self.notes += f", geodetic standardisation added"
        self.match_geo_transform = transform
        return self

    def match_standardise_enu(self) -> Self:
        """
        Normalises ENU projections and tracks
        """
        # Scan all training set
        e_max = max([flight.match["Trk_E"].max() for flight in self])
        e_min = min([flight.match["Trk_E"].min() for flight in self])
        n_max = max([flight.match["Trk_N"].max() for flight in self])
        n_min = min([flight.match["Trk_N"].min() for flight in self])
        u_max = max([flight.match["Trk_U"].max() for flight in self])
        u_min = min([flight.match["Trk_U"].min() for flight in self])
        steps_max = max([len(flight.match) for flight in self])

        # Store the dictionary of the transformation
        transform = {
            "e_max": float(e_max),
            "e_min": float(e_min),
            "n_max": float(n_max),
            "n_min": float(n_min),
            "u_max": float(u_max),
            "u_min": float(u_min),
            "steps_max": steps_max,
        }
        for flight in self.flights:
            flight.match_enu_transform(transform)

        self.notes += f", ENU standardisation added"
        self.match_enu_transform = transform
        return self

    def plot(
        self,
        wide=140,
        ratio=0.6,
        move_east=-70,
        move_north=0,
        alpha=0.1,
        cmap="brg",
        title="",
    ) -> None:
        """
        Plot a scene
        """
        Fl.plot_very_long_list(
            flights=self.flights,
            title=title,
            wide=wide,
            alpha=alpha,
            cmap=cmap,
            move_east=move_east,
            move_north=move_north,
            ratio=ratio,
        )

    def plot_elevation_profile(
        self,
        xlim=500,
        ylim=None,
        factortime=1,
        endalign=True,
        alpha=0.1,
    ) -> None:
        """
        Plots the elevation profile of a scene
        """
        Fl.plot_elevation_profile(
            flights=self.flights,
            xlim=xlim,
            ylim=ylim,
            factortime=factortime,
            endalign=endalign,
            alpha=alpha,
        )

    def plot_elevation_histogram(
        self,
        bins=100,
        alpha_threshold=0,
        xlim=None,
        ylim=None,
        endalign=True,
        cmap="prism_r",
    ) -> None:
        """
        Plots the elevation profile of a scene as histogram
        """
        Fl.plot_elevation_histogram(
            flights=self.flights,
            bins=bins,
            alpha_threshold=alpha_threshold,
            xlim=xlim,
            ylim=ylim,
            endalign=endalign,
            cmap=cmap,
        )

    #
    # Other dunders
    #
    def __iter__(self) -> Iterator[Fl]:
        return iter(self.flights)

    def __getitem__(self, key: int) -> Fl:
        return self.flights[key]

    def __repr__(self) -> str:
        return f"Scene with {len(self.flights)} flights, {self.notes}."
