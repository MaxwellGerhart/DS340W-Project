"""Public analytical tool interface for future model-driven tool calling."""
from .analytics import window_metrics, summarize

class MatchTools:
    def __init__(self, events, tracking=None):
        from .data import validate_events, validate_tracking
        self.events=validate_events(events)
        if len({e['match_id'] for e in self.events})!=1:raise ValueError('MatchTools requires one match.')
        self.tracking=validate_tracking(tracking or [],self.events)
        self.match_id=self.events[0]['match_id']
        self.teams=sorted({e['team'] for e in self.events})

    def _window(self,team,start_minute,end_minute):
        if team not in self.teams or start_minute<0 or end_minute<=start_minute:
            raise ValueError('Known team and a nonempty, nonnegative interval are required.')
        return window_metrics(self.events,self.tracking,team,start_minute*60,end_minute*60)

    def calculate_pitch_control(self,team,start_minute,end_minute):
        w=self._window(team,start_minute,end_minute)
        return {'method':'nearest-player Voronoi proxy, not calibrated pitch control',
                'available':bool(w['n_frames']),'surface':w['surface'],'n_frames':w['n_frames'],
                'spatial_control':w['metrics'].get('spatial_control'),'central_control':w['metrics'].get('central_control')}

    def calculate_team_compactness(self,team,start_minute,end_minute):
        w=self._window(team,start_minute,end_minute)
        return {'n_frames':w['n_frames'],'width_m':w['metrics'].get('defensive_width_m'),
                'depth_m':w['metrics'].get('defensive_depth_m'),
                'method':'Outfield width/depth. Goalkeeper excluded by nearest-own-goal heuristic. Not phase-conditioned.'}

    def build_passing_network(self,team,start_minute,end_minute):
        return self._window(team,start_minute,end_minute)['network']

    def find_change_points(self,window_minutes=15):
        return summarize(self.events,self.tracking,window_minutes)['change_points']

    def compare_game_periods(self,team,period_a,period_b):
        a,b=self._window(team,*period_a),self._window(team,*period_b)
        return {'team':team,'period_a':period_a,'period_b':period_b,'differences':{
            key:b['metrics'][key]-value for key,value in a['metrics'].items()
            if value is not None and b['metrics'].get(key) is not None}}

    def retrieve_supporting_events(self,team=None,start_minute=0,end_minute=200,event_type=None):
        if team is not None and team not in self.teams:raise ValueError('Unknown team.')
        return [e for e in self.events if start_minute*60<=e['time_seconds']<end_minute*60
                and (team is None or e['team']==team) and (event_type is None or e['type']==event_type)]
