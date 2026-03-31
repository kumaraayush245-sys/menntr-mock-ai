import { apiClient } from './client';

export interface ConversationMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface Interview {
  id: number;
  user_id: number;
  resume_id: number | null;
  title: string;
  status: 'pending' | 'in_progress' | 'completed';
  conversation_history: ConversationMessage[] | null;
  resume_context: Record<string, unknown> | null;
  job_description: string | null;
  feedback: Record<string, unknown> | null;
  turn_count: number;
  current_message: string | null;
  sandbox: Record<string, unknown> | null;
  show_code_editor: boolean;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface SkillDetail {
  score: number;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
}

export interface InterviewSkillBreakdown {
  interview_id: number;
  interview_title: string;
  completed_at: string | null;
  skill_breakdown: {
    communication: SkillDetail;
    technical: SkillDetail;
    problem_solving: SkillDetail;
    code_quality: SkillDetail;
  };
}

export interface SkillComparisonResponse {
  comparison: Record<string, unknown>;
  interviews: Array<{ id: number; title: string; completed_at: string | null }>;
}

export const interviewsApi = {
  list: async (): Promise<Interview[]> => {
    return apiClient.get<Interview[]>('/api/v1/interviews');
  },

  get: async (id: number): Promise<Interview> => {
    return apiClient.get<Interview>(`/api/v1/interviews/${id}`);
  },

  create: async (data: {
    title: string;
    resume_id?: number;
    job_description?: string;
  }): Promise<Interview> => {
    return apiClient.post<Interview>('/api/v1/interviews', data);
  },

  start: async (id: number): Promise<Interview> => {
    return apiClient.post<Interview>('/api/v1/interviews/start', { interview_id: id });
  },

  respond: async (id: number, message: string): Promise<Interview> => {
    return apiClient.post<Interview>('/api/v1/interviews/respond', {
      interview_id: id,
      message,
    });
  },

  complete: async (id: number): Promise<Interview> => {
    return apiClient.post<Interview>('/api/v1/interviews/complete', { interview_id: id });
  },

  submitCode: async (id: number, code: string, language = 'python'): Promise<Interview> => {
    return apiClient.post<Interview>('/api/v1/interviews/submit-code', {
      interview_id: id,
      code,
      language,
    });
  },

  delete: async (id: number): Promise<void> => {
    return apiClient.delete<void>(`/api/v1/interviews/${id}`);
  },

  getFeedback: async (id: number): Promise<Record<string, unknown>> => {
    return apiClient.get(`/api/v1/interviews/${id}/feedback`);
  },

  getInterviewSkills: async (id: number): Promise<InterviewSkillBreakdown> => {
    return apiClient.get<InterviewSkillBreakdown>(`/api/v1/interviews/${id}/skills`);
  },

  getUserAnalytics: async (): Promise<Record<string, unknown>> => {
    return apiClient.get('/api/v1/interviews/analytics/user');
  },

  getSkillProgression: async (): Promise<Record<string, unknown>> => {
    return apiClient.get('/api/v1/interviews/analytics/skills/progression');
  },

  getSkillAverages: async (): Promise<Record<string, unknown>> => {
    return apiClient.get('/api/v1/interviews/analytics/skills/averages');
  },

  compareSkills: async (interviewIds: number[]): Promise<SkillComparisonResponse> => {
    return apiClient.get<SkillComparisonResponse>(
      `/api/v1/interviews/analytics/skills/compare?interview_ids=${interviewIds.join(',')}`
    );
  },
};
