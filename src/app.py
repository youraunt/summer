# import pandas as pd


# album_data = pd.read_csv('data/spotify_albums.csv')
# artist_data = pd.read_csv('data/spotify_artists.csv')
# track_data = pd.read_csv('data/spotify_tracks.csv')



# ## join artist genre information and album release date with track dataset
# # drop irrelevant columns
# # get only tracks after 1990
# def join_genre_and_date(artist_df, album_df, track_df):
#     album = album_df.rename(columns={'id':"album_id"}).set_index('album_id')
#     artist = artist_df.rename(columns={'id':"artists_id",'name':"artists_name"}).set_index('artists_id')
#     track = track_df.set_index('album_id').join(album['release_date'], on='album_id' )
#     track.artists_id = track.artists_id.apply(lambda x: x[2:-2])
#     track = track.set_index('artists_id').join(artist[['artists_name','genres']], on='artists_id' )
#     track.reset_index(drop=False, inplace=True)
#     track['release_year'] = pd.to_datetime(track.release_date).dt.year
#     track.drop(columns = ['Unnamed: 0','country','track_name_prev','track_number','type'], inplace = True)
    
#     return track[track.release_year >= 1990]

# def get_filtered_track_df(df, genres_to_include):
#     df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
#     df_exploded = df.explode("genres")[df.explode("genres")["genres"].isin(genres_to_include)]
#     df_exploded.loc[df_exploded["genres"]=="korean pop", "genres"] = "k-pop"
#     df_exploded_indices = list(df_exploded.index.unique())
#     df = df[df.index.isin(df_exploded_indices)]
#     df = df.reset_index(drop=True)
#     return df

# genres_to_include = genres = ['dance pop', 'electronic', 'electropop', 'hip hop', 'jazz', 'k-pop', 'latin', 'pop', 'pop rap', 'r&b', 'rock']
# track_with_year_and_genre = join_genre_and_date(artist_data, album_data, track_data)
# filtered_track_df = get_filtered_track_df(track_with_year_and_genre, genres_to_include)


# filtered_track_df["uri"] = filtered_track_df["uri"].str.replace("spotify:track:", "")
# filtered_track_df = filtered_track_df.drop(columns=['analysis_url', 'available_markets'])

# filtered_track_df.to_csv("filtered_track_df.csv", index=False)

import streamlit as st
st.set_page_config(page_title="Song Recommendation", layout="wide")

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("filtered_track_df.csv")
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios


title = "Song Recommendation Engine"
st.title(title)

st.write("First of all, welcome! This is the place where you can customize what you want to listen to based on genre and several key audio features. Try playing around with different settings and listen to the songs recommended by our system!")
st.markdown("##")

with st.container():
    col1, col2,col3,col4 = st.columns((2,0.5,0.5,0.5))
    with col3:
        st.markdown("***Choose your genre:***")
        genre = st.radio(
            "",
            genre_names, index=genre_names.index("Pop"))
    with col1:
        st.markdown("***Choose features to customize:***")
        start_year, end_year = st.slider(
            'Select the year range',
            1990, 2019, (2015, 2019)
        )
        acousticness = st.slider(
            'Acousticness',
            0.0, 1.0, 0.5)
        danceability = st.slider(
            'Danceability',
            0.0, 1.0, 0.5)
        energy = st.slider(
            'Energy',
            0.0, 1.0, 0.5)
        instrumentalness = st.slider(
            'Instrumentalness',
            0.0, 1.0, 0.0)
        valence = st.slider(
            'Valence',
            0.0, 1.0, 0.45)
        tempo = st.slider(
            'Tempo',
            0.0, 244.0, 118.0)

tracks_per_page = 6
test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)

tracks = []
for uri in uris:
    track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
    tracks.append(track)

if 'previous_inputs' not in st.session_state:
    st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat

current_inputs = [genre, start_year, end_year] + test_feat
if current_inputs != st.session_state['previous_inputs']:
    if 'start_track_i' in st.session_state:
        st.session_state['start_track_i'] = 0
    st.session_state['previous_inputs'] = current_inputs

if 'start_track_i' not in st.session_state:
    st.session_state['start_track_i'] = 0

with st.container():
    col1, col2, col3 = st.columns([2,1,2])
    if st.button("Recommend More Songs"):
        if st.session_state['start_track_i'] < len(tracks):
            st.session_state['start_track_i'] += tracks_per_page

    current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    if st.session_state['start_track_i'] < len(tracks):
        for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
            if i%2==0:
                with col1:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                        r=audio[:5],
                        theta=audio_feats[:5]))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)
        
            else:
                with col3:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

    else:
        st.write("No songs left to recommend")