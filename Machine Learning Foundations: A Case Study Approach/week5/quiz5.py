import turicreate as tc


song_data = tc.SFrame("song_data.sframe")
artists = ['Kanye West', 'Foo Fighters', 'Taylor Swift', 'Lady GaGa']
print(song_data)

for artist in artists:
    song_artist = song_data[song_data['artist'] == artist]
    artist_user = song_artist['user_id'].unique()
    print(artist + ':' + str(len(artist_user)))

grouped = song_data.groupby(key_column_names='artist', operations={'total_count': tc.aggregate.SUM('listen_count')})
print(grouped.sort('total_count', ascending=False))
print(grouped.sort('total_count', ascending=True))

