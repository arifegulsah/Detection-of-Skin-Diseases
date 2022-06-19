using DeriHastaliklari.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using System.Linq;
using System;
using Microsoft.EntityFrameworkCore;
using System.Collections.Generic;

namespace DeriHastaliklari.Controllers
{
    public class DataGoruntuleme : Controller
    {
        Context c = new Context();


        [Authorize]
        public IActionResult Index()
        {
            List<PatientList> model = new List<PatientList>();
            return View(model);
        }

       
        public async Task<IActionResult> PatientSearch(PatientFilter request)
        {

            var query = (from p in c.Patients
                        join ap in c.AddPhotos
                        on p.PatientId equals ap.PatientId
                        where
                            p.TcNo == request.TcNo
                            || p.Name == request.Name
                            || p.Surname == request.Surname
                        select new PatientList()
                        {
                            TcNo = p.TcNo,
                            Name = p.Name,
                            Surname = p.Surname,
                            Medicine = p.Medicine,
                            MedicineText = p.Medicine ? "Evet" : "Hayır", // bool olduğu için küçük bir if kontrolü kullanıcıya true false yerine daha anlamlı bir ifade göstermek için
                            Allergy = p.Allergy,
                            AllergyText = p.Allergy ? "Evet" : "Hayır",
                            Disease = p.Disease,
                            DiseaseText = p.Disease ? "Evet" : "Hayır",
                            Itch = ap.Itch ? "Evet" : "Hayır",
                            Pain = ap.Pain ? "Evet" : "Hayır",
                            Cont = ap.Cont ? "Evet" : "Hayır",
                            Additional = ap.Additional
                        });

            if (request.Allergy != "0")
            {
                query = query.Where(x=>x.Allergy == Convert.ToBoolean(request.Allergy));
            }


            if (request.Medicine != "0")
            {
                query = query.Where(x => x.Medicine == Convert.ToBoolean(request.Medicine));
            }

            if (request.Disease != "0")
            {
                query = query.Where(x => x.Disease == Convert.ToBoolean(request.Disease));
            }

            var data = await query.ToListAsync();


            return View("Index", data);
        }

       
        
    }
}
