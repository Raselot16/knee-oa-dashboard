"""Interactive scientific dashboard; use existing trained project artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard_core import FEATURES, MODEL_NAMES, LABEL_NAMES, RED_FLAGS, SOURCES, load_context, assess

BASE=Path(__file__).resolve().parent
COLORS=['#6685ad','#137e83','#c6ced8']
st.set_page_config(page_title='Knee OA Observatory',page_icon='🦵',layout='wide')
st.markdown('''<style>
.stApp{background:#f4f7fa;color:#19334a}.block-container{max-width:1240px;padding:4.5rem 2rem 3rem}
[data-testid="stSidebar"]{background:#eaf0f5}h1,h2,h3{color:#16364d;letter-spacing:-.025em}
.eyebrow{font-size:.76rem;color:#16777d;letter-spacing:.16em;font-weight:700;margin-bottom:.4rem}
.intro{color:#526b7e;max-width:850px;font-size:1.03rem;line-height:1.7}
[data-testid="stMetric"]{padding:1rem;background:white;border:1px solid #dce5ec;border-radius:12px}
[data-testid="stMetricValue"]{color:#17485e}[data-testid="stForm"]{background:white;border-color:#dce5ec;border-radius:14px}
.stTabs [data-baseweb="tab-list"]{gap:1.4rem}button[kind="primary"]{background:#14777c;border-color:#14777c}
.foot{border-top:1px solid #dce5ec;margin-top:2rem;padding-top:1rem;color:#698091;font-size:.83rem}
</style>''',unsafe_allow_html=True)


@st.cache_resource(show_spinner='Đang đọc dữ liệu và mô hình…')
def cached_context(base,signature):
    return load_context(base)


def chart(fig,key,height=320):
    fig.update_layout(height=height,margin=dict(t=25,b=45,l=35,r=20),paper_bgcolor='white',plot_bgcolor='white',
                      font=dict(family='Arial, sans-serif',color='#29465d',size=12),legend=dict(orientation='h',y=-.22,x=0))
    fig.update_xaxes(showgrid=False,automargin=True)
    fig.update_yaxes(gridcolor='#edf1f5',zeroline=False,automargin=True)
    st.plotly_chart(fig,width='stretch',theme=None,key=key,config={'displayModeBar':False})


def histogram(counts,edges,xlabel,key,color):
    e=np.array(edges)
    fig=go.Figure(go.Bar(x=(e[:-1]+e[1:])/2,y=counts,width=np.diff(e)*.92,marker_color=color,
                         customdata=list(zip(e[:-1],e[1:])),
                         hovertemplate='Khoảng %{customdata[0]:g}–%{customdata[1]:g}<br>Số hồ sơ: %{y}<extra></extra>'))
    fig.update_layout(xaxis_title=xlabel,yaxis_title='Số hồ sơ')
    chart(fig,key)


with st.sidebar:
    st.markdown('### Knee OA Observatory')
    st.caption('DỮ LIỆU · MÔ HÌNH · SỨC KHỎE')
st.markdown('<div class="eyebrow">CLINICAL DATA / MACHINE LEARNING</div>',unsafe_allow_html=True)
st.title('Khám phá sức khỏe khớp gối')
st.markdown('<p class="intro">Biểu đồ từ dữ liệu dự án, điểm mô hình cho hồ sơ cá nhân và hướng dẫn chăm sóc có nguồn tham khảo.</p>',unsafe_allow_html=True)
st.info('Bản nghiên cứu trên dữ liệu MatchThem dẫn xuất có chỉnh sửa. Mô hình chưa xác định mức độ bệnh, nguy cơ loãng xương hoặc nguy cơ trong 48 tháng.')
try:
    paths=list(BASE.glob('*.csv'))+list(BASE.glob('dashboard_data.json'))+list((BASE/'results').rglob('*.json'))+list((BASE/'results').rglob('*.joblib'))+list((BASE/'results').rglob('split.csv'))
    sig=tuple(sorted((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths))
    ctx=cached_context(str(BASE),sig)
except Exception as exc:
    st.error(f'Chưa thể mở dashboard: {exc}')
    st.caption('Đặt các file giao diện trong thư mục dự án có CSV và results, hoặc dùng gói web_deploy được chuẩn bị từ máy đã huấn luyện.')
    st.stop()
s=ctx['summary']; report=ctx['report']
with st.sidebar:
    st.divider();st.markdown('**Mô hình đang dùng**');st.write(MODEL_NAMES[ctx['model_name']])
    st.caption('Chọn theo ROC-AUC kiểm định chéo trong train.' if ctx['report_kind']=='comparison' else 'Mô hình cơ sở.')
    st.markdown('**Phạm vi tham chiếu**')
    for key,label in [('age','Tuổi'),('bmi','BMI')]:
        st.write(f"{label}: {s['input_ranges'][key][0]:g}–{s['input_ranges'][key][1]:g}")
    st.caption('Điểm mô hình và hướng dẫn theo triệu chứng là hai phần độc lập.')
overview,personal,evaluation,method=st.tabs(['Tổng quan dữ liệu','Đánh giá cá nhân','Hiệu năng mô hình','Phương pháp & nguồn'])

with overview:
    keys=['all','labeled'] if 'all' in s['datasets'] else ['labeled']
    selected=st.selectbox('Bộ dữ liệu hiển thị',keys,format_func=lambda k:s['datasets'][k]['title'],key='overview_source')
    d=s['datasets'][selected]
    cols=st.columns(4)
    for col,label,value in zip(cols,['Tổng hồ sơ','Có nhãn kết cục','Tuổi trung vị','BMI trung vị'],
                               [f"{d['n']:,}",f"{sum(d['label_counts'][:2]):,}",f"{d['age_median']:.0f} tuổi",f"{d['bmi_median']:.1f}"]):
        col.metric(label,value)
    st.caption('Thống kê mô tả bộ dữ liệu dẫn xuất; không đại diện tỷ lệ mắc bệnh của dân số. Ô thiếu được loại khỏi biểu đồ số tương ứng và thống kê riêng.')
    a,b=st.columns(2,gap='large')
    with a:
        st.subheader('Phân bố nhãn dữ liệu')
        fig=go.Figure(go.Bar(x=LABEL_NAMES,y=d['label_counts'],marker_color=COLORS,text=d['label_counts'],textposition='outside'))
        fig.update_layout(yaxis_title='Số hồ sơ');chart(fig,'labels')
        st.caption('Nhãn 0 = “at risk”; nhãn 1 = “diagnosed”. Nhãn 0 không có nghĩa hoàn toàn khỏe mạnh.')
    with b:
        st.subheader('Phân bố tuổi');histogram(d['age_hist'],d['age_edges'],'Tuổi (năm)','ages',COLORS[0])
    a,b=st.columns(2,gap='large')
    with a:
        st.subheader('Phân bố BMI');histogram(d['bmi_hist'],d['bmi_edges'],'BMI (kg/m²)','bmis',COLORS[1])
        st.caption(f"Hồ sơ có BMI: {d['n']-d['missing'].get('bmi',0):,}. BMI = cân nặng (kg) / [chiều cao (m)]².")
    with b:
        st.subheader('Phân bố đồng thời tuổi và BMI')
        al=[f'{int(x)}–{int(y)-1}' for x,y in zip(d['age_edges'][:-1],d['age_edges'][1:])]
        bl=[f'{x:g}–<{y:g}' for x,y in zip(d['bmi_edges'][:-1],d['bmi_edges'][1:])]
        fig=go.Figure(go.Heatmap(z=np.array(d['heatmap']).T,x=al,y=bl,colorscale='Teal',colorbar=dict(title='Số hồ sơ'),
                                 hovertemplate='Tuổi %{x}<br>BMI %{y}<br>Số hồ sơ: %{z}<extra></extra>'))
        fig.update_layout(xaxis_title='Nhóm tuổi (năm)',yaxis_title='Khoảng BMI (kg/m²)');chart(fig,'age_bmi')
        st.caption(f"Có {d['paired_n']:,} hồ sơ đủ cả hai biến. Màu thể hiện số hồ sơ, không thể hiện mức nguy hiểm.")
    a,b=st.columns(2,gap='large')
    with a:
        st.subheader('Tỷ lệ nhãn 1 theo nhóm tuổi')
        g=d['age_groups']
        rates=[100*v['label_1_n']/v['labeled_n'] if v['labeled_n'] else None for v in g]
        fig=go.Figure(go.Bar(x=[v['group'] for v in g],y=rates,marker_color=COLORS[1],
                             customdata=[[v['label_1_n'],v['labeled_n']] for v in g],
                             hovertemplate='Tuổi %{x}<br>Nhãn 1: %{customdata[0]}/%{customdata[1]}<br>Tỷ lệ %{y:.1f}%<extra></extra>'))
        fig.update_layout(yaxis_title='Nhãn 1 / hồ sơ có nhãn (%)',yaxis_range=[0,100],xaxis_title='Nhóm tuổi (năm)');chart(fig,'age_rate')
        st.caption('Mẫu số là hồ sơ có nhãn trong từng nhóm tuổi. Liên hệ mô tả không chứng minh nhân quả.')
    with b:
        st.subheader('Dữ liệu thiếu')
        missing={k:v for k,v in d['missing'].items() if v}
        if missing:
            fig=go.Figure(go.Bar(x=list(missing.values()),y=list(missing),orientation='h',marker_color='#9baab9',text=list(missing.values()),textposition='outside'))
            fig.update_layout(xaxis_title='Số ô thiếu');chart(fig,'missing')
        else:st.info('Bảng đang chọn không có ô thiếu.')

with personal:
    st.subheader('Một hồ sơ, hai góc nhìn')
    st.caption('Bốn đặc trưng được đưa vào mô hình. Các câu hỏi triệu chứng chỉ dùng cho hướng dẫn đi khám.')
    mode=st.radio('Cách nhập BMI',['Nhập BMI có sẵn','Tính từ chiều cao và cân nặng'],horizontal=True,key='bmi_mode')
    left,right=st.columns([1,1.1],gap='large')
    with left:
        with st.form('profile_form'):
            st.markdown('#### 1. Đặc trưng cho mô hình')
            age=st.number_input('Tuổi (năm)',min_value=18,max_value=100,value=None,step=1,key='age',placeholder='Ví dụ: 68')
            if mode=='Nhập BMI có sẵn':
                bmi=st.number_input('BMI (kg/m²)',min_value=10.,max_value=70.,value=None,step=.1,key='bmi',placeholder='Ví dụ: 28.1')
            else:
                height=st.number_input('Chiều cao (cm)',min_value=100.,max_value=230.,value=None,step=.5,key='height')
                weight=st.number_input('Cân nặng (kg)',min_value=20.,max_value=250.,value=None,step=.1,key='weight')
                bmi=None if height is None or weight is None else weight/(height/100)**2
                st.caption('BMI tính được hiển thị sau khi gửi.')
            smoke=st.selectbox('Tình trạng hút thuốc',['Chưa chọn','Không','Có'],key='smoking')
            bone=st.selectbox('Tình trạng loãng xương đã được xác định',['Chưa biết / chưa chọn','Không','Có'],key='osteoporosis')
            st.caption('Loãng xương là đầu vào đã biết; mô hình này không tự chẩn đoán loãng xương.')
            st.markdown('#### 2. Triệu chứng để định hướng đi khám')
            flags=st.multiselect('Chọn các dấu hiệu hiện có',RED_FLAGS,key='red_flags')
            st.caption('Nếu đang có dấu hiệu trong danh sách, hãy liên hệ cơ sở y tế ngay để được hướng dẫn; không cần chờ nhập đủ biểu mẫu.')
            checked=st.checkbox('Tôi đã kiểm tra danh sách dấu hiệu ở trên.',key='symptoms_checked')
            prolonged=st.selectbox('Đau gối không cải thiện sau vài tuần?',['Chưa trả lời','Không','Có'],key='persistent')
            submitted=st.form_submit_button('Xem đánh giá và hướng dẫn',type='primary',width='stretch')
        if submitted:
            st.session_state.pop('assessment',None)
            inputs={'age':age,'bmi':bmi,'smoking':None if smoke=='Chưa chọn' else int(smoke=='Có'),
                    'osteoporosis_baseline':None if bone=='Chưa biết / chưa chọn' else int(bone=='Có')}
            try:
                r=assess(ctx,inputs,flags,checked,None if prolonged=='Chưa trả lời' else prolonged=='Có')
                r['time_utc']=datetime.now(timezone.utc).isoformat();st.session_state['assessment']=r
            except Exception as exc:st.error(f'Chưa thể xử lý hồ sơ: {exc}')
    with right:
        r=st.session_state.get('assessment')
        if r and r.get('model_sha256')!=s['model_sha256']:
            st.session_state.pop('assessment',None);r=None
        if not r:
            with st.container(border=True):
                st.markdown('### Kết quả sẽ hiển thị tại đây')
                st.write('Nhập hồ sơ và nhấn **Xem đánh giá và hướng dẫn**.')
                st.caption('Hồ sơ ngoài phạm vi dữ liệu sẽ không được dự đoán bằng ML.')
        else:
            st.caption('Kết quả của lần gửi gần nhất; nhấn lại nút sau khi thay đổi thông tin. Hồ sơ đã gửi được ghi bên dưới.')
            care=r['care'];st.markdown('#### Hướng dẫn từ triệu chứng bạn khai báo')
            display={'urgent':st.error,'appointment':st.warning,'selfcare':st.info,'unknown':st.info}[care['level']]
            display('**'+care['title']+'**\n\n'+care['message'])
            if care['basis']:st.caption('Căn cứ: '+'; '.join(care['basis']))
            st.caption(f"Nguồn: [NHS – đau gối]({care['source']}). Phần này độc lập với điểm mô hình.")
            st.markdown('#### Điểm tương đối của mô hình')
            pred=r['prediction']
            if pred:
                a,b=st.columns(2);a.metric('Điểm mô hình (0–1)',f"{pred['score']:.3f}");b.metric('Nhãn dự đoán',str(pred['label']))
                group=pred['relative_group']
                if group:
                    st.markdown('### '+group['name'])
                    st.caption('Ba nhóm theo tam phân vị điểm tập train; không tương ứng với tốt / có bệnh / nguy hiểm.')
                    ref=s['score_reference'];e=np.array(ref['edges'])
                    fig=go.Figure(go.Bar(x=(e[:-1]+e[1:])/2,y=ref['hist'],width=np.diff(e)*.9,marker_color='#aec9d5'))
                    for q in group['cutoffs']:fig.add_vline(x=q,line_dash='dot',line_color='#8b9daa')
                    fig.add_vline(x=pred['score'],line_color='#b37422',line_width=3)
                    fig.update_layout(xaxis_title='Điểm mô hình',yaxis_title='Số hồ sơ train',xaxis_range=[0,1]);chart(fig,'personal_score',260)
                    st.caption('Vạch vàng: hồ sơ vừa nhập. Vạch chấm: mốc chia nhóm. Điểm tham chiếu được tính trên chính tập train, không phải đánh giá độc lập.')
                else:st.info('Chưa có phân bố điểm train phù hợp để chia nhóm tương đối.')
                st.caption('Điểm chưa được hiệu chỉnh thành nguy cơ bệnh; nhãn 0/1 không phải mức độ nặng.')
            else:
                for issue in r['issues']:st.warning(issue)
            if care['level']!='urgent':
                st.markdown('#### Gợi ý chăm sóc để tham khảo')
                st.markdown('- Vận động phù hợp và tăng dần; trao đổi với bác sĩ hoặc chuyên viên phục hồi chức năng về bài tập an toàn.\n- Nếu đang thừa cân, trao đổi kế hoạch quản lý cân nặng để giảm tải lên khớp.')
                st.caption(f"Nguồn: [NIAMS – chăm sóc khớp]({SOURCES['Chăm sóc khớp · NIAMS']}). Gợi ý không được chọn theo nhóm điểm ML.")
                if r['inputs'].get('smoking')==1:st.write('Bạn khai đang hút thuốc: có thể tìm hỗ trợ để bỏ thuốc.')
                if r['inputs'].get('osteoporosis_baseline')==1:st.write('Bạn khai đã được xác định loãng xương: trao đổi bài tập phù hợp và phòng ngã với nhân viên y tế.')
                if r['inputs'].get('smoking')==1 or r['inputs'].get('osteoporosis_baseline')==1:
                    st.caption(f"Nguồn: [NIAMS – chăm sóc loãng xương]({SOURCES['Loãng xương · NIAMS']}).")
            with st.expander('Hồ sơ đã dùng trong lần gửi này'):
                st.json(r['inputs']);st.caption('Triệu chứng không nằm trong bốn cột đầu vào của mô hình.')
            st.download_button('Tải kết quả thử nghiệm (.json)',json.dumps(r,ensure_ascii=False,indent=2),file_name='knee_oa_assessment_demo.json',mime='application/json')

with evaluation:
    st.subheader('Đánh giá trên dữ liệu chưa dùng để fit')
    if ctx['report_kind']=='comparison':
        st.caption(f"Train: {report['train_rows']:,} · Test: {report['test_rows']:,}. Chọn mô hình bằng kiểm định chéo trong train.")
        measures=[('accuracy','Accuracy'),('precision_class_1','Precision lớp 1'),('recall_class_1','Recall lớp 1'),('f1_class_1','F1 lớp 1'),('roc_auc','ROC-AUC'),('average_precision','Average Precision')]
        table={MODEL_NAMES[n]:{title:f"{v['metrics'][k]:.4f}" for k,title in measures} for n,v in report['test'].items()}
        st.dataframe(pd.DataFrame(table),width='stretch')
        fig=go.Figure()
        for i,(name,v) in enumerate(report['test'].items()):fig.add_bar(name=MODEL_NAMES[name],x=[t for _,t in measures],y=[v['metrics'][k] for k,_ in measures],marker_color=COLORS[i])
        fig.update_layout(barmode='group',yaxis_title='Giá trị chỉ số',yaxis_range=[0,1]);chart(fig,'model_metrics',350)
        cols=st.columns(2)
        for col,(name,v) in zip(cols,report['test'].items()):
            with col:
                st.markdown('**'+MODEL_NAMES[name]+'**')
                fig=go.Figure(go.Heatmap(z=v['confusion_matrix'],x=['Dự đoán 0','Dự đoán 1'],y=['Thực tế 0','Thực tế 1'],text=v['confusion_matrix'],texttemplate='%{text}',colorscale='Blues',showscale=False))
                fig.update_yaxes(autorange='reversed');chart(fig,'cm_'+name,270)
        cv=[{'Mô hình':MODEL_NAMES[n],'ROC-AUC trung bình':v['roc_auc']['mean'],'Độ lệch chuẩn':v['roc_auc']['std']} for n,v in report['cv'].items()]
        st.dataframe(pd.DataFrame(cv).round(4),hide_index=True,width='stretch')
        st.caption('Độ lệch chuẩn giữa 5 fold không phải khoảng tin cậy. Chênh lệch nhỏ chưa chứng minh vượt trội; không tinh chỉnh theo test.')
    else:
        st.dataframe(pd.DataFrame([report.get('metrics',{})]).T,width='stretch');st.info('Chạy compare_models.py để có đánh giá hai mô hình.')

with method:
    st.subheader('Ba loại thông tin cần phân biệt')
    st.dataframe(pd.DataFrame([
        ['Biểu đồ dữ liệu','Thống kê từ CSV','Không phải tỷ lệ mắc bệnh của dân số'],
        ['Nhóm điểm tương đối','Tam phân vị điểm trên tập train','Không phải ngưỡng nguy cơ hay chẩn đoán'],
        ['Hướng dẫn đi khám','Triệu chứng tự khai và tài liệu NHS','Độc lập với mô hình ML; không thay thế khám'],
    ],columns=['Thành phần','Căn cứ','Giới hạn']),hide_index=True,width='stretch')
    st.markdown('''
**Bài toán:** phân loại nhị phân `koa_followup` từ tuổi, BMI, hút thuốc và loãng xương ban đầu. Nhãn nguồn 0 = “at risk”, 1 = “diagnosed” tại lần theo dõi; chưa xác minh thời hạn kết cục.

**Loãng xương và thoái hóa khớp gối là hai tình trạng khác nhau.** Loãng xương hiện là đầu vào đã biết. Muốn dự đoán loãng xương cần dữ liệu, nhãn và đánh giá riêng; đánh giá y tế có thể bao gồm đo mật độ xương DXA.

**Giới hạn:** bộ MatchThem là dẫn xuất OAI có chỉnh sửa. Mã giới tính không khớp tài liệu nên chưa dùng. Chưa có ID OAI được xác minh để bảo đảm chia tập độc lập theo người. Điểm trên bộ này không đại diện nguy cơ bệnh trong cộng đồng.

**Phương pháp:** chia train/test trước; điền thiếu và chuẩn hóa trong pipeline. Ba nhóm điểm chỉ mô tả thứ hạng, không thay đổi bài toán phân loại nhị phân hoặc các chỉ số đã báo cáo.
''')
    if s.get('score_reference'):
        ref=s['score_reference'];lo,hi=ref['tertiles']
        st.write(f"Mốc tam phân vị: {lo:.4f} và {hi:.4f}, từ {ref['n']:,} hồ sơ train. Nhóm thấp: điểm < mốc 1; trung gian: mốc 1 ≤ điểm < mốc 2; cao: điểm ≥ mốc 2.")
    st.markdown('**Tài liệu tham khảo**')
    for title,url in SOURCES.items():st.markdown(f'- [{title}]({url})')
    st.caption('Nguồn y khoa kiểm tra ngày 27/10/2026. Thông tin giáo dục, không phải phác đồ điều trị cá nhân.')
st.markdown('<div class="foot">Knee OA Observatory · Dashboard học thuật · Điểm mô hình, triệu chứng và chẩn đoán là các khái niệm riêng biệt.</div>',unsafe_allow_html=True)
