using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using DeriHastaliklari.Models;
using System.Web;
using System.Linq;
using System.Collections.Generic;
using System.Security.Claims;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;
using System.IO;
using System;

namespace DeriHastalikleri.Controllers
{
    public class PatientController : Controller
    {

        Context c = new Context();
        /* private readonly Context _context;

         public PatientController(//Context context
             )
         {
             //PatientController üzerine gelip Generate Constructor yap otomatik gelyor
            // _context = context;
         }*/
        public IActionResult PatientLogin()
        { 
        //    var user = c.Patients.FirstOrDefault(x => x.Email == email && x.Password == password);
        //    if (user != null)
        //    {
        //        HttpContext.Session.SetInt32("id", user.PatientId);
        //        HttpContext.Session.SetString("fullname", user.Name + " " + user.Surname);
        //        return RedirectToAction("Photo", "Patient");
        //    }
        //    else
        //    {
        //        ViewData["Message"] = "Böyle bir kullanıcı yok!";
        //    }
            return View(); // Redirect yapılabilir
        }

        [HttpPost]
        public IActionResult PatientLogin2(string email, string password)
        {
            var user = c.Patients.FirstOrDefault(x => x.Email == email && x.Password == password);
            if (user != null)
            {
                HttpContext.Session.SetInt32("id", user.PatientId);
                HttpContext.Session.SetString("fullname", user.Name + " " + user.Surname);
                return RedirectToAction("Photo", "Patient");
            }
            else
            {
                ViewData["Message"] = "Böyle bir kullanıcı yok!";
            }

            return View("PatientLogin");
        }
        public IActionResult PatientRegister() { return View(); }
        public async Task<IActionResult> PatientRegisterSave(Patient p)
        {
           
            await c.Patients.AddAsync(p);
            await c.SaveChangesAsync();
            return RedirectToAction("PatientLogin", "Patient");

        }

        [HttpGet]
        public IActionResult DoctorLogin()
        {
            return View();
        }

        //alt+shift+f10
        //ctrl+k+c
        //ctrl+k+u
        public async Task<IActionResult> DoctorLogin(Doctor request)
        {
            var bilgiler = c.Doctors.FirstOrDefault(x => x.Username == request.Username && x.Password == request.Password);
            if (bilgiler != null)
            {
                //kullanıcı bilgilerini tutar 
                var claims = new List<Claim>
                {
                    new Claim(ClaimTypes.Name, request.Username)
                };

                var useridentity = new ClaimsIdentity(claims, "Patient");
                ClaimsPrincipal principal = new ClaimsPrincipal(useridentity);
                await HttpContext.SignInAsync(principal);

                return RedirectToAction("Index", "DataGoruntuleme");

            }

            return View();
        }


        public IActionResult PForgotPassword()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> PForgotPassword(PForgotPassword request)
        {
            var bilgiler = await c.Patients.FirstOrDefaultAsync(x => x.Name == request.Name && x.Surname == request.Surname && x.TcNo == request.TcNo && x.bthDay == request.bthDay && x.Email == request.Email);

            if (bilgiler != null)
            {
                ViewData["PatientId"] = bilgiler.PatientId;
                return View("PNewPassword");
            }

            return View("PForgotPassword");
        }

        [HttpPost]
        public async Task<IActionResult> PNewPassword(NewPassword request)
        {

            if (request.NewPass != request.NewPassRepeat)
            {
                ViewData["warning"] = "Parolanız uyuşmamaktadır!";

                return View("PNewPassword");
            }
            else
            {
                var bilgiler = await c.Patients.FirstOrDefaultAsync(x => x.PatientId == request.PatientId);
                bilgiler.Password = request.NewPass; //eski passworde yenisini set ediyoruz.
                c.SaveChanges();
                return View("PatientLogin");
            }
        }


        public IActionResult Photo()
        {
            return View();
        }



        [HttpPost]
        public async Task<IActionResult> Photo(AddPhoto a)
        {
          
            if (a.ImageURL != null)
            {
                var extension = Path.GetExtension(a.ImageURL.FileName);
                if (extension.Contains("jpg") || extension.Contains("png") || extension.Contains("jpeg"))
                {
                    
                    a.PatientId = HttpContext.Session.GetInt32("id").Value;
                    c.AddPhotos.Add(a);
                    await c.SaveChangesAsync();
                    var newImagename = a.ImageId + extension;
                    var location = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot/resimler/", newImagename);
                    var stream = new FileStream(location, FileMode.Create);
                    a.ImageURL.CopyTo(stream);
                }
                else
                {
                    ViewData["warning"] = "Lütfen resim formatında seçim yapınız.";
                    return View();
                }
                
                
            }
            // <img src="/wwwroot/resimler/3.jpg"
            return View();

        }

    }
}
